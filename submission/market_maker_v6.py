# Paste this class below the official challenge model definitions.
# Keep the official lines 1-260 unchanged.

class MarketMaker:
    """V6 probability market maker: competitive RFQ, informed-flow FOK.

    Trade-sign convention used by the challenge template:
      quantity > 0: this market maker buys / becomes long
      quantity < 0: this market maker sells / becomes short

    FOK convention:
      OrderType.BUY: the customer buys, so this market maker sells
      OrderType.SELL: the customer sells, so this market maker buys
    """

    # Competitive initial values.  Live controls are still learned from
    # rolling history, fills, information markouts and realised P&L.
    BASE_HALF_SPREAD = 0.012
    MIN_BASE_HALF_SPREAD = 0.01
    MAX_BASE_HALF_SPREAD = 0.04
    MAX_HALF_SPREAD = 0.075
    CLEAN_RFQ_MAX_HALF_SPREAD = 0.035
    MODEL_ERROR_MULTIPLIER = 0.15
    TENOR_MULTIPLIER = 0.001
    TOXICITY_MULTIPLIER = 0.35
    PORTFOLIO_RISK_AVERSION = 0.03
    MAX_VOLATILITY_SPREAD = 0.01
    MIN_VOLATILITY_OBSERVATIONS = 3

    CASH_BUFFER_FRACTION = 0.15
    MAX_FOK_LOSS_FRACTION = 0.15
    QUOTE_RISK_FRACTION = 0.10
    PER_OPTION_CONTRACT_CAP = 200
    UNDERLYING_CONTRACT_CAP = 300
    GROSS_CONTRACT_CAP = 500
    MAX_QUOTE_SIZE = 30

    FOK_BASE_EDGE = 0.008
    FOK_MODEL_ERROR_MULTIPLIER = 0.12
    FOK_MAX_REQUIRED_EDGE = 0.06
    FOK_MAX_INFORMATION_LOGIT_SHIFT = 2.25

    MARKET_WINDOW = 60
    VOLATILITY_WINDOW = 30
    TRADE_WINDOW = 30
    PNL_WINDOW = 20
    FLOW_WINDOW = 30
    MARKET_HALF_LIFE = 20.0
    TARGET_FILL_RATE = 0.30
    PRIOR_SAMPLE_SIZE = 30.0

    # Reduced path budgets preserve the runtime configuration that cleared the
    # long hidden sessions; state-level caches prevent repeated simulation.
    LIVE_FALLBACK_PATHS = 1_000
    THEO_PATHS = 5_000
    COVARIANCE_PATHS = 500

    # Set this to True only for the three VERBOSE debugging sessions.
    DEBUG = False
    DEBUG_LIMIT = 200

    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.position: Position = Position()

        # Estimated model and calibration information.
        self.estimated_parameters = None
        self.history_steps: int = 1
        self.rate_history: list[float] = []
        self.ajr_history: list[float] = []
        self.thr_history: list[float] = []

        # Our own copy of the autograder's maximum-loss cash accounting.
        self.initial_cash_balance: float = cash_balance
        self.trade_lots_by_option_id = defaultdict(list)

        # Flow/adverse-selection state.
        self.pending_markouts: list[
            tuple[BinaryOption, float, int, int, float]
        ] = []
        # Counterparty markout is the direction-adjusted movement in fair value
        # after a trade, not execution P&L.  Negative values mean that the fair
        # value moved in the customer's favour.
        self.counterparty_markout = defaultdict(float)
        self.counterparty_observations = defaultdict(int)
        self.counterparty_markout_by_key = defaultdict(float)
        self.counterparty_observations_by_key = defaultdict(int)
        self.flow_signal_by_key = defaultdict(float)
        self.flow_learning_history_by_key = defaultdict(list)
        self.learned_flow_coefficient_by_key = defaultdict(float)
        self.counterparty_requests_this_step = defaultdict(int)
        self.fok_directional_requests_this_step = defaultdict(int)

        # Rolling execution feedback. Markouts are per contract; P&L values are
        # signed monetary totals. They start empty because burn-in contains no
        # executions.
        self.markout_history: list[float] = []
        self.markout_pnl_history: list[float] = []
        self.realized_pnl_history: list[float] = []
        self.fill_rate_history: list[float] = []
        # Only non-expiry fair changes are used as quote-volatility evidence,
        # and they are kept by payoff exposure rather than pooled globally.
        self.live_fair_change_history_by_key = defaultdict(list)
        self.adaptive_base_half_spread: float = self.BASE_HALF_SPREAD
        self.cumulative_realized_pnl: float = 0.0
        self.peak_realized_equity: float = cash_balance
        self.order_opportunities_this_step: int = 0
        self.fills_this_step: int = 0
        self._debug_count: int = 0

        # Prices do not depend on inventory, so a fill must not invalidate them.
        # These caches are cleared only after the market state or fitted model
        # changes.  Binary payoff samples are stored as integer bit sets: one
        # bit per Monte Carlo path.  Cov(X,Y) can then be calculated using a
        # fast bitwise AND and bit_count instead of another simulation.
        self.live_price_cache: dict[BinaryOption, float] = {}
        self.covariance_scenarios_by_step: dict[
            int, list[dict[int, float]]
        ] = {}
        self.covariance_scenario_max_steps: int = -1
        self.covariance_payoff_bits: dict[BinaryOption, int] = {}

    def _clear_market_state_caches(self) -> None:
        """Invalidate only computations that depend on state/model values."""
        self.live_price_cache.clear()
        self.covariance_scenarios_by_step.clear()
        self.covariance_scenario_max_steps = -1
        self.covariance_payoff_bits.clear()

    # ------------------------------------------------------------------
    # State updates and trade bookkeeping
    # ------------------------------------------------------------------

    def on_step_advance(
        self,
        new_underlying_state: list[Underlying],
        new_option_state: list[BinaryOption],
    ) -> None:
        final_values = {
            underlying.underlying_id: underlying.value
            for underlying in new_underlying_state
        }

        realized_pnl_this_step = 0.0

        # Settle our internal maximum-loss cash account for options expiring now.
        for option in self.active_option_state:
            if option.steps_until_expiry != 1:
                continue

            payoff = option.expiry_valuation(final_values)
            for quantity, trade_price in self.trade_lots_by_option_id.pop(
                option.option_id, []
            ):
                realized_pnl_this_step += quantity * (
                    payoff - trade_price
                )
                if quantity > 0:
                    # Long contract: receive its 0/1 payoff.
                    self.cash_balance += quantity * payoff
                elif quantity < 0:
                    # Short contract: recover unused worst-case collateral.
                    self.cash_balance += (-quantity) * (1.0 - payoff)

            self.position.option_quantity_by_option_id.pop(option.option_id, None)

        # Drawdown is based on settled economic P&L.  Reserved collateral is
        # handled separately by risk utilisation and is not counted as a loss.
        self.cumulative_realized_pnl += realized_pnl_this_step
        realized_equity = (
            self.initial_cash_balance + self.cumulative_realized_pnl
        )
        self.peak_realized_equity = max(
            self.peak_realized_equity, realized_equity
        )

        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state
        self._clear_market_state_caches()

        # Add the new day to the rolling market window. At the start of
        # execution these lists contain only however many burn-in days were
        # actually provided; they are never assumed to contain 60 or 200 days.
        self.rate_history.append(
            final_values[FED_FUNDS_RATE_UNDERLYING_ID]
        )
        self.ajr_history.append(final_values[AJARAI_UNDERLYING_ID])
        self.thr_history.append(final_values[THERIODIC_UNDERLYING_ID])
        self.rate_history = self.rate_history[-self.MARKET_WINDOW :]
        self.ajr_history = self.ajr_history[-self.MARKET_WINDOW :]
        self.thr_history = self.thr_history[-self.MARKET_WINDOW :]

        if len(self.rate_history) >= 2:
            self._estimate_parameters_from_history()

        # One-step markout: did the fair value move against us after the trade?
        new_options_by_id = {
            option.option_id: option for option in new_option_state
        }

        markout_pnl_this_step = 0.0

        for (
            old_option,
            price,
            quantity,
            counterparty_id,
            fair_at_trade,
        ) in self.pending_markouts:
            new_option = new_options_by_id.get(old_option.option_id)

            contract_survived = (
                new_option is not None
                and new_option.steps_until_expiry > 0
            )

            if contract_survived:
                new_fair = self.price_option(new_option)
            elif old_option.steps_until_expiry == 1:
                new_fair = old_option.expiry_valuation(final_values)
            else:
                # Defensive fallback if an active contract is unexpectedly absent.
                continue

            fair_change = new_fair - fair_at_trade

            if quantity > 0:
                # We bought: a rise in fair value is favourable information.
                information_markout = fair_change
                execution_markout = new_fair - price
            else:
                # We sold: a rise in fair value is adverse information.
                information_markout = -fair_change
                execution_markout = price - new_fair

            markout_pnl_this_step += (
                abs(quantity) * execution_markout
            )
            self.markout_history.append(information_markout)

            risk_key = self._option_risk_key(old_option)

            # Resolution to 0/1 is useful for P&L and toxicity, but it is not a
            # reusable volatility observation for unrelated live contracts.
            if contract_survived:
                volatility_history = (
                    self.live_fair_change_history_by_key[risk_key]
                )
                volatility_history.append(fair_change)
                del volatility_history[:-self.VOLATILITY_WINDOW]

            # Positive flow means the customer bought. Learn whether that flow
            # predicts an upward subsequent change in fair value.
            normalized_client_flow = self._clip(
                -quantity / 5.0, -2.0, 2.0
            )
            flow_history = self.flow_learning_history_by_key[risk_key]
            flow_history.append((normalized_client_flow, fair_change))
            del flow_history[:-self.FLOW_WINDOW]

            observations = self.counterparty_observations[counterparty_id]
            old_markout = self.counterparty_markout[counterparty_id]
            alpha = 1.0 if observations == 0 else 0.20
            self.counterparty_markout[counterparty_id] = (
                (1.0 - alpha) * old_markout
                + alpha * information_markout
            )
            self.counterparty_observations[counterparty_id] += 1

            counterparty_key = (counterparty_id, risk_key)
            key_observations = self.counterparty_observations_by_key[
                counterparty_key
            ]
            old_key_markout = self.counterparty_markout_by_key[
                counterparty_key
            ]
            key_alpha = 1.0 if key_observations == 0 else 0.20
            self.counterparty_markout_by_key[counterparty_key] = (
                (1.0 - key_alpha) * old_key_markout
                + key_alpha * information_markout
            )
            self.counterparty_observations_by_key[counterparty_key] += 1

        self.pending_markouts.clear()

        self.markout_history = self.markout_history[-self.TRADE_WINDOW :]
        self.markout_pnl_history.append(markout_pnl_this_step)
        self.markout_pnl_history = self.markout_pnl_history[
            -self.PNL_WINDOW :
        ]
        self.realized_pnl_history.append(realized_pnl_this_step)
        self.realized_pnl_history = self.realized_pnl_history[
            -self.PNL_WINDOW :
        ]

        fill_rate = self.fills_this_step / max(
            self.order_opportunities_this_step, 1
        )
        self.fill_rate_history.append(fill_rate)
        self.fill_rate_history = self.fill_rate_history[-self.PNL_WINDOW :]

        self._update_adaptive_controls()

        self.order_opportunities_this_step = 0
        self.fills_this_step = 0

        # Old flow should gradually lose predictive importance, separately for
        # each payoff exposure.  Reset same-step request counts as well.
        for risk_key in list(self.flow_signal_by_key):
            self.flow_signal_by_key[risk_key] *= 0.50
        self.counterparty_requests_this_step.clear()
        self.fok_directional_requests_this_step.clear()

        self._debug(
            "step_advance",
            cash=round(self.cash_balance, 4),
            realized_pnl=round(realized_pnl_this_step, 4),
            cumulative_realized_pnl=round(
                self.cumulative_realized_pnl, 4
            ),
            base_half_spread=round(
                self.adaptive_base_half_spread, 6
            ),
        )

    def on_trade(
        self,
        option: BinaryOption,
        price: float,
        quantity: int,
        counterparty_id: int,
    ) -> None:
        self.position.add_option_quantity(option.option_id, quantity)
        fair_at_trade = self.price_option(option)

        # Mirror the challenge's maximum-loss solvency accounting.
        if quantity > 0:
            self.cash_balance -= quantity * price
        elif quantity < 0:
            self.cash_balance -= (-quantity) * (1.0 - price)

        self.trade_lots_by_option_id[option.option_id].append((quantity, price))
        self.pending_markouts.append(
            (option, price, quantity, counterparty_id, fair_at_trade)
        )
        self.fills_this_step += 1

        # Positive signal means customers have recently been buying contracts
        # with the same underlying payoff exposure.
        client_signed_quantity = -quantity
        risk_key = self._option_risk_key(option)
        self.flow_signal_by_key[risk_key] = self._clip(
            0.80 * self.flow_signal_by_key[risk_key]
            + client_signed_quantity / 5.0,
            -6.0,
            6.0,
        )
        # A trade changes quantities, but not option values or the joint payoff
        # distribution.  Keeping the state caches here is the main runtime win.

        self._debug(
            "trade",
            option_id=option.option_id,
            price=price,
            signed_quantity=quantity,
            counterparty=counterparty_id,
            cash=self.cash_balance,
            position=self.position.option_quantity_by_option_id.get(
                option.option_id, 0
            ),
        )

    # Some versions of the visible template call this callback ``trade``,
    # while the current checker dispatches fills to ``on_trade``.  Keep this
    # compatibility wrapper so the same class works with either interface.
    def trade(
        self,
        option: BinaryOption,
        price: float,
        quantity: int,
        counterparty_id: int,
    ) -> None:
        self.on_trade(option, price, quantity, counterparty_id)

    # ------------------------------------------------------------------
    # Required display name
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # type: ignore[empty-body]
        return "ChuanjieAdaptiveMMv6"

    # ------------------------------------------------------------------
    # Small mathematical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    @staticmethod
    def _logistic(value: float) -> float:
        """Numerically stable enough for the bounded V6 logit shifts."""
        if value >= 0.0:
            exponential = math.exp(-value)
            return 1.0 / (1.0 + exponential)
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def _debug(self, event: str, **values) -> None:
        """Bounded VERBOSE-test logging; disabled for scored submissions."""
        if not self.DEBUG or self._debug_count >= self.DEBUG_LIMIT:
            return
        self._debug_count += 1
        try:
            print("[MM DEBUG]", event, values, flush=True)
        except Exception:
            pass

    @staticmethod
    def _option_risk_key(option: BinaryOption):
        """Group flow/volatility only across the same payoff exposure."""
        return tuple(
            (
                leg.underlying_id,
                round(leg.weight, 8),
            )
            for leg in option.legs
        )

    def _current_values(self) -> dict[int, float]:
        return {
            underlying.underlying_id: underlying.value
            for underlying in self.underlying_state
        }

    def _rate_distribution(
        self,
        parameters: MarketParameters,
        initial_rate: float,
        steps: int,
    ) -> dict[float, float]:
        distribution: dict[float, float] = {initial_rate: 1.0}

        for _ in range(steps):
            next_distribution = defaultdict(float)

            for rate, probability in distribution.items():
                up_probability, down_probability = (
                    parameters.tilted_rate_probabilities(rate)
                )
                flat_probability = 1.0 - up_probability - down_probability

                up_rate = parameters.next_rate_value(rate, 1)
                down_rate = parameters.next_rate_value(rate, -1)

                next_distribution[up_rate] += probability * up_probability
                next_distribution[down_rate] += probability * down_probability
                next_distribution[rate] += probability * flat_probability

            distribution = dict(next_distribution)

        return distribution

    # ------------------------------------------------------------------
    # Fast analytic pricing for the option types used in scored trading
    # ------------------------------------------------------------------

    def _analytic_price(
        self,
        parameters: MarketParameters,
        option: BinaryOption,
    ):
        values = self._current_values()
        steps = option.steps_until_expiry

        if steps == 0:
            return option.expiry_valuation(values)

        initial_rate = values[FED_FUNDS_RATE_UNDERLYING_ID]
        terminal_rate_distribution = self._rate_distribution(
            parameters, initial_rate, steps
        )
        weights = {
            leg.underlying_id: leg.weight for leg in option.legs
        }

        # A single FED leg can be priced exactly from its Markov distribution.
        if (
            len(weights) == 1
            and FED_FUNDS_RATE_UNDERLYING_ID in weights
        ):
            weight = weights[FED_FUNDS_RATE_UNDERLYING_ID]
            return sum(
                probability
                for terminal_rate, probability
                in terminal_rate_distribution.items()
                if weight * terminal_rate >= option.strike
            )

        # A single AJR or THR leg is conditionally lognormal given terminal FED.
        if len(weights) == 1 and (
            AJARAI_UNDERLYING_ID in weights
            or THERIODIC_UNDERLYING_ID in weights
        ):
            company_id = next(iter(weights))
            weight = weights[company_id]

            if company_id == AJARAI_UNDERLYING_ID:
                current_value = values[company_id]
                drift = parameters.ajarai_drift
                rate_beta = parameters.ajarai_rate_beta
                idio_std = parameters.ajarai_idio_std_dev
                sector_beta = parameters.ajarai_sector_beta
            else:
                current_value = values[company_id]
                drift = parameters.theriodic_drift
                rate_beta = parameters.theriodic_rate_beta
                idio_std = parameters.theriodic_idio_std_dev
                sector_beta = parameters.theriodic_sector_beta

            variance = steps * (
                idio_std ** 2
                + (sector_beta * parameters.sector_std_dev) ** 2
            )
            threshold = option.strike / weight
            price = 0.0

            for terminal_rate, rate_probability in (
                terminal_rate_distribution.items()
            ):
                mean = (
                    math.log(current_value)
                    + steps * drift
                    + rate_beta * (terminal_rate - initial_rate)
                )

                if weight > 0:
                    if threshold <= 0:
                        conditional_probability = 1.0
                    elif variance <= 1e-16:
                        conditional_probability = float(
                            mean >= math.log(threshold)
                        )
                    else:
                        z_score = (
                            math.log(threshold) - mean
                        ) / math.sqrt(variance)
                        conditional_probability = 1.0 - self._normal_cdf(
                            z_score
                        )
                else:
                    if threshold <= 0:
                        conditional_probability = 0.0
                    elif variance <= 1e-16:
                        conditional_probability = float(
                            mean <= math.log(threshold)
                        )
                    else:
                        z_score = (
                            math.log(threshold) - mean
                        ) / math.sqrt(variance)
                        conditional_probability = self._normal_cdf(z_score)

                price += rate_probability * conditional_probability

            return price

        # AJR-versus-THR spread with strike zero.
        if (
            len(weights) == 2
            and AJARAI_UNDERLYING_ID in weights
            and THERIODIC_UNDERLYING_ID in weights
            and abs(option.strike) < 1e-12
        ):
            ajr_weight = weights[AJARAI_UNDERLYING_ID]
            thr_weight = weights[THERIODIC_UNDERLYING_ID]

            if ajr_weight * thr_weight < 0:
                log_ratio_variance = steps * (
                    parameters.ajarai_idio_std_dev ** 2
                    + parameters.theriodic_idio_std_dev ** 2
                    + (
                        (
                            parameters.ajarai_sector_beta
                            - parameters.theriodic_sector_beta
                        )
                        * parameters.sector_std_dev
                    ) ** 2
                )
                price = 0.0

                for terminal_rate, rate_probability in (
                    terminal_rate_distribution.items()
                ):
                    log_ratio_mean = (
                        math.log(
                            values[AJARAI_UNDERLYING_ID]
                            / values[THERIODIC_UNDERLYING_ID]
                        )
                        + steps
                        * (
                            parameters.ajarai_drift
                            - parameters.theriodic_drift
                        )
                        + (
                            parameters.ajarai_rate_beta
                            - parameters.theriodic_rate_beta
                        )
                        * (terminal_rate - initial_rate)
                    )

                    if ajr_weight > 0:
                        threshold = math.log(-thr_weight / ajr_weight)
                        if log_ratio_variance <= 1e-16:
                            conditional_probability = float(
                                log_ratio_mean >= threshold
                            )
                        else:
                            conditional_probability = 1.0 - self._normal_cdf(
                                (threshold - log_ratio_mean)
                                / math.sqrt(log_ratio_variance)
                            )
                    else:
                        threshold = math.log(thr_weight / -ajr_weight)
                        if log_ratio_variance <= 1e-16:
                            conditional_probability = float(
                                log_ratio_mean <= threshold
                            )
                        else:
                            conditional_probability = self._normal_cdf(
                                (threshold - log_ratio_mean)
                                / math.sqrt(log_ratio_variance)
                            )

                    price += rate_probability * conditional_probability

                return price

        return None

    # ------------------------------------------------------------------
    # Exact-dynamics deterministic Monte Carlo, including daily rounding
    # ------------------------------------------------------------------

    def _advance_values_once(
        self,
        values: dict[int, float],
        parameters: MarketParameters,
        rng: random.Random,
    ) -> dict[int, float]:
        old_rate = values[FED_FUNDS_RATE_UNDERLYING_ID]
        up_probability, down_probability = (
            parameters.tilted_rate_probabilities(old_rate)
        )

        uniform_draw = rng.random()
        if uniform_draw < up_probability:
            new_rate = parameters.next_rate_value(old_rate, 1)
        elif uniform_draw < up_probability + down_probability:
            new_rate = parameters.next_rate_value(old_rate, -1)
        else:
            new_rate = old_rate

        rate_change = round(new_rate - old_rate, 2)
        sector_shock = rng.gauss(0.0, parameters.sector_std_dev)

        ajr_log_return = (
            parameters.ajarai_drift
            + parameters.ajarai_rate_beta * rate_change
            + parameters.ajarai_sector_beta * sector_shock
            + rng.gauss(0.0, parameters.ajarai_idio_std_dev)
        )
        thr_log_return = (
            parameters.theriodic_drift
            + parameters.theriodic_rate_beta * rate_change
            + parameters.theriodic_sector_beta * sector_shock
            + rng.gauss(0.0, parameters.theriodic_idio_std_dev)
        )

        return {
            FED_FUNDS_RATE_UNDERLYING_ID: new_rate,
            AJARAI_UNDERLYING_ID: round(
                values[AJARAI_UNDERLYING_ID] * math.exp(ajr_log_return),
                2,
            ),
            THERIODIC_UNDERLYING_ID: round(
                values[THERIODIC_UNDERLYING_ID] * math.exp(thr_log_return),
                2,
            ),
        }

    def _monte_carlo_price(
        self,
        parameters: MarketParameters,
        option: BinaryOption,
        number_of_paths: int,
    ) -> float:
        initial_values = self._current_values()

        if option.steps_until_expiry == 0:
            return option.expiry_valuation(initial_values)

        seed = repr(
            (
                parameters,
                option,
                tuple(sorted(initial_values.items())),
                number_of_paths,
            )
        )
        rng = random.Random(seed)
        successful_paths = 0.0

        for _ in range(number_of_paths):
            values = dict(initial_values)
            for _ in range(option.steps_until_expiry):
                values = self._advance_values_once(values, parameters, rng)
            successful_paths += option.expiry_valuation(values)

        return successful_paths / number_of_paths

    # ------------------------------------------------------------------
    # Required pricing methods
    # ------------------------------------------------------------------

    def price_option(self, option: BinaryOption) -> float:  # type: ignore[empty-body]
        cached_price = self.live_price_cache.get(option)
        if cached_price is not None:
            return cached_price

        if self.estimated_parameters is None:
            return 0.5

        price = self._analytic_price(self.estimated_parameters, option)
        if price is None:
            price = self._monte_carlo_price(
                self.estimated_parameters,
                option,
                self.LIVE_FALLBACK_PATHS,
            )

        price = self._clip(price, 0.0, 1.0)
        self.live_price_cache[option] = price
        return price

    def price_option_from_parameters(  # type: ignore[empty-body]
        self,
        market_parameters: MarketParameters,
        option: BinaryOption,
    ) -> float:
        # FED is genuinely discrete and can be priced exactly.
        if (
            len(option.legs) == 1
            and option.legs[0].underlying_id
            == FED_FUNDS_RATE_UNDERLYING_ID
        ):
            price = self._analytic_price(market_parameters, option)
        else:
            # This follows the supplied dynamics exactly, including cent rounding.
            price = self._monte_carlo_price(
                market_parameters,
                option,
                self.THEO_PATHS,
            )

        return self._clip(price, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Portfolio payoff covariance and inventory reservation price
    # ------------------------------------------------------------------

    def _ensure_covariance_scenarios(self, required_steps: int) -> None:
        """Generate one common set of market paths for the current state.

        An older version generated a fresh joint simulation after every fill because
        position quantities were included in the covariance-cache key.  Market
        scenarios do not depend on our inventory.  Generate them once and reuse
        them for every candidate/held-option pair until the market advances.
        """
        if self.estimated_parameters is None:
            return

        # Build only as far as the options currently entering the inventory
        # calculation.  An unrelated long-dated active option should not make
        # every short-dated quote simulate its entire horizon.
        maximum_steps = max(required_steps, 0)

        if self.covariance_scenario_max_steps >= maximum_steps:
            return

        initial_values = self._current_values()
        seed = repr(
            (
                "v6-shared-covariance-scenarios",
                self.estimated_parameters,
                tuple(sorted(initial_values.items())),
                self.COVARIANCE_PATHS,
                maximum_steps,
            )
        )
        rng = random.Random(seed)
        scenarios_by_step = {
            step: [] for step in range(maximum_steps + 1)
        }

        for _ in range(self.COVARIANCE_PATHS):
            values = dict(initial_values)
            scenarios_by_step[0].append(values)

            for step in range(1, maximum_steps + 1):
                values = self._advance_values_once(
                    values, self.estimated_parameters, rng
                )
                scenarios_by_step[step].append(values)

        self.covariance_scenarios_by_step = scenarios_by_step
        self.covariance_scenario_max_steps = maximum_steps
        self.covariance_payoff_bits.clear()

    def _covariance_payoff_bitset(self, option: BinaryOption) -> int:
        """Return the option's sampled binary payoffs as a compact bit set."""
        cached_bits = self.covariance_payoff_bits.get(option)
        if cached_bits is not None:
            return cached_bits

        self._ensure_covariance_scenarios(option.steps_until_expiry)
        scenarios = self.covariance_scenarios_by_step[
            option.steps_until_expiry
        ]

        payoff_bits = 0
        for path_index, values in enumerate(scenarios):
            if option.expiry_valuation(values) > 0.5:
                payoff_bits |= 1 << path_index

        self.covariance_payoff_bits[option] = payoff_bits
        return payoff_bits

    def _portfolio_payoff_covariance(
        self,
        candidate: BinaryOption,
    ) -> float:
        if self.estimated_parameters is None:
            return 0.0

        active_by_id = {
            option.option_id: option for option in self.active_option_state
        }
        held: list[tuple[BinaryOption, int]] = []

        for option_id, quantity in (
            self.position.option_quantity_by_option_id.items()
        ):
            option = active_by_id.get(option_id)
            if quantity != 0 and option is not None:
                held.append((option, quantity))

        if not held:
            return 0.0

        self._ensure_covariance_scenarios(
            max(
                candidate.steps_until_expiry,
                *(option.steps_until_expiry for option, _ in held),
            )
        )
        candidate_bits = self._covariance_payoff_bitset(candidate)
        candidate_mean = (
            candidate_bits.bit_count() / self.COVARIANCE_PATHS
        )
        portfolio_covariance = 0.0

        for held_option, quantity in held:
            held_bits = self._covariance_payoff_bitset(held_option)
            held_mean = held_bits.bit_count() / self.COVARIANCE_PATHS
            cross_mean = (
                (candidate_bits & held_bits).bit_count()
                / self.COVARIANCE_PATHS
            )
            payoff_covariance = cross_mean - candidate_mean * held_mean
            portfolio_covariance += quantity * payoff_covariance

        return portfolio_covariance

    def _counterparty_toxicity(
        self,
        counterparty_id: int,
        option: BinaryOption,
    ) -> float:
        """Blend exposure-specific toxicity with a smaller global prior."""
        # Negative information markout indicates adverse selection.  Both
        # estimates are shrunk so one lucky or unlucky fill cannot dominate.
        global_mean = self.counterparty_markout[counterparty_id]
        global_observations = self.counterparty_observations[counterparty_id]
        global_shrinkage = global_observations / (
            global_observations + 10.0
        )
        global_toxicity = max(
            -global_mean * global_shrinkage, 0.0
        )

        counterparty_key = (
            counterparty_id,
            self._option_risk_key(option),
        )
        specific_mean = self.counterparty_markout_by_key[counterparty_key]
        specific_observations = self.counterparty_observations_by_key[
            counterparty_key
        ]
        specific_shrinkage = specific_observations / (
            specific_observations + 5.0
        )
        specific_toxicity = max(
            -specific_mean * specific_shrinkage, 0.0
        )

        return self._clip(
            0.25 * global_toxicity + 0.75 * specific_toxicity,
            0.0,
            0.10,
        )

    def _recent_fair_volatility(self, option: BinaryOption) -> float:
        history = self.live_fair_change_history_by_key.get(
            self._option_risk_key(option), []
        )
        observations = len(history)
        if observations < self.MIN_VOLATILITY_OBSERVATIONS:
            return 0.0
        raw_rms = math.sqrt(
            sum(change ** 2 for change in history) / observations
        )
        # Sparse live markouts are shrunk toward zero.  Expiry jumps never enter
        # this history, and even a noisy bucket cannot add more than two cents
        # of half-spread through the volatility channel.
        shrinkage = observations / (observations + 8.0)
        return self._clip(shrinkage * raw_rms, 0.0, 0.06)

    def _model_standard_error(self, fair_value: float) -> float:
        effective_history = max(self.history_steps, 25)
        return math.sqrt(
            max(fair_value * (1.0 - fair_value), 0.01)
            / effective_history
        )

    def _recent_negative_pnl_fraction(self) -> float:
        # Realized P&L is definitive. Markout P&L is an earlier but noisier
        # signal, so give it smaller weight to avoid double-counting a trade.
        recent_pnl = sum(self.realized_pnl_history) + 0.25 * sum(
            self.markout_pnl_history
        )
        return max(
            0.0,
            -recent_pnl / max(self.initial_cash_balance, 1e-12),
        )

    def _drawdown_fraction(self) -> float:
        realised_equity = (
            self.initial_cash_balance + self.cumulative_realized_pnl
        )
        return self._clip(
            (self.peak_realized_equity - realised_equity)
            / max(self.initial_cash_balance, 1e-12),
            0.0,
            1.0,
        )

    def _risk_utilisation(self) -> float:
        # Our cash account falls when maximum-loss collateral is reserved.
        return self._clip(
            1.0
            - self.cash_balance / max(self.initial_cash_balance, 1e-12),
            0.0,
            1.0,
        )

    def _dynamic_risk_aversion(self) -> float:
        multiplier = (
            1.0
            + 2.0 * self._drawdown_fraction()
            + 0.75 * self._risk_utilisation()
            + 2.0 * self._recent_negative_pnl_fraction()
        )
        return self._clip(
            self.PORTFOLIO_RISK_AVERSION * multiplier,
            0.01,
            0.10,
        )

    def _dynamic_cash_buffer_fraction(self) -> float:
        return self._clip(
            self.CASH_BUFFER_FRACTION
            + 0.15 * self._drawdown_fraction()
            + 0.10 * self._risk_utilisation()
            + 0.10 * self._recent_negative_pnl_fraction(),
            self.CASH_BUFFER_FRACTION,
            0.45,
        )

    def _dynamic_risk_scale(self) -> float:
        # Collateral utilisation and settled losses reduce new risk.  Collateral
        # is not also treated as realised drawdown, avoiding double-counting.
        return self._clip(
            1.0
            - 0.50 * self._drawdown_fraction()
            - 0.35 * self._risk_utilisation()
            - 0.75 * self._recent_negative_pnl_fraction(),
            0.40,
            1.0,
        )

    def _update_adaptive_controls(self) -> None:
        # Learn the predictive coefficient in
        #     next fair change = intercept + theta * customer flow + noise.
        # Keep the regressions separated by payoff exposure; THR buying should
        # not mechanically move a FED-only quote.
        for risk_key, history in self.flow_learning_history_by_key.items():
            observations = len(history)
            if observations < 4:
                continue
            mean_flow = sum(
                flow for flow, _change in history
            ) / observations
            mean_change = sum(
                change for _flow, change in history
            ) / observations
            denominator = sum(
                (flow - mean_flow) ** 2
                for flow, _change in history
            )
            if denominator > 1e-12:
                raw_coefficient = sum(
                    (flow - mean_flow) * (change - mean_change)
                    for flow, change in history
                ) / denominator
                shrinkage = observations / (observations + 20.0)
                self.learned_flow_coefficient_by_key[risk_key] = self._clip(
                    shrinkage * raw_coefficient,
                    -0.03,
                    0.03,
                )

        average_markout = (
            sum(self.markout_history) / len(self.markout_history)
            if self.markout_history
            else 0.0
        )
        average_fill_rate = (
            sum(self.fill_rate_history) / len(self.fill_rate_history)
            if self.fill_rate_history
            else 0.0
        )

        # Negative markouts and excessive fills widen the base spread; safe
        # markouts and very low fills narrow it gradually. Recent losses add a
        # smaller extra defensive pressure.
        spread_change = (
            0.04 * (-average_markout)
            + 0.003 * (average_fill_rate - self.TARGET_FILL_RATE)
            + 0.005 * self._recent_negative_pnl_fraction()
        )
        self.adaptive_base_half_spread = self._clip(
            self.adaptive_base_half_spread + spread_change,
            self.MIN_BASE_HALF_SPREAD,
            self.MAX_BASE_HALF_SPREAD,
        )

    def _quote_components(
        self,
        option: BinaryOption,
        counterparty_id: int,
    ) -> tuple[float, float, float, float, float]:
        fair_value = self.price_option(option)

        portfolio_covariance = self._portfolio_payoff_covariance(option)
        portfolio_skew = self._clip(
            self._dynamic_risk_aversion() * portfolio_covariance,
            -0.12,
            0.12,
        )

        risk_key = self._option_risk_key(option)

        # Customer buying pressure raises the centre slightly; selling lowers
        # it.  Signals are local to the same payoff exposure.
        flow_adjustment = self._clip(
            self.learned_flow_coefficient_by_key[risk_key]
            * self.flow_signal_by_key[risk_key],
            -0.02,
            0.02,
        )

        reservation_price = self._clip(
            fair_value - portfolio_skew + flow_adjustment,
            0.0,
            1.0,
        )

        model_standard_error = self._model_standard_error(fair_value)
        toxicity = self._counterparty_toxicity(counterparty_id, option)
        tenor = min(math.sqrt(option.steps_until_expiry), 3.0)
        live_volatility = self._recent_fair_volatility(option)
        volatility_spread = min(
            self.MAX_VOLATILITY_SPREAD,
            0.20 * live_volatility,
        )
        repeat_count = max(
            self.counterparty_requests_this_step[counterparty_id] - 1,
            0,
        )
        repeat_penalty = min(
            0.010,
            0.002 * repeat_count,
        )
        # Extreme probabilities receive only a small price penalty; their
        # asymmetric payoff risk is handled mainly through side-specific size.
        tail_penalty = 0.005 * self._clip(
            (abs(fair_value - 0.50) - 0.30) / 0.20,
            0.0,
            1.0,
        )

        half_spread = (
            self.adaptive_base_half_spread
            + self.MODEL_ERROR_MULTIPLIER * model_standard_error
            + self.TENOR_MULTIPLIER * tenor
            + self.TOXICITY_MULTIPLIER * toxicity
            + volatility_spread
            + repeat_penalty
            + tail_penalty
        )

        # Ordinary central-probability RFQs are the primary source of spread
        # capture.  Cap their width so that overlapping uncertainty terms do
        # not make us systematically lose to a competent fixed-width dealer.
        clean_rfq = (
            0.15 <= fair_value <= 0.85
            and toxicity <= 0.015
            and repeat_count == 0
            and abs(portfolio_skew) <= 0.04
            and self._risk_utilisation() <= 0.60
        )
        if clean_rfq:
            half_spread = min(
                half_spread, self.CLEAN_RFQ_MAX_HALF_SPREAD
            )

        half_spread = self._clip(
            half_spread, self.MIN_BASE_HALF_SPREAD, self.MAX_HALF_SPREAD
        )

        return (
            fair_value,
            reservation_price,
            half_spread,
            toxicity,
            portfolio_skew,
        )

    # ------------------------------------------------------------------
    # Hard position, exposure and cash limits
    # ------------------------------------------------------------------

    def _risk_limits(self) -> tuple[int, int, int]:
        # Contract counts are emergency concentration guards only.  Bankruptcy
        # protection is price-sensitive and is enforced through maximum dollar
        # loss and the cash buffer.
        return (
            self.PER_OPTION_CONTRACT_CAP,
            self.UNDERLYING_CONTRACT_CAP,
            self.GROSS_CONTRACT_CAP,
        )

    def _position_capacity(
        self,
        option: BinaryOption,
        direction: int,
        requested_cap: int,
    ) -> int:
        """Maximum safe signed-direction quantity under hard position limits."""
        per_option_limit, underlying_limit, gross_limit = self._risk_limits()
        active_by_id = {
            active.option_id: active for active in self.active_option_state
        }
        candidate_underlyings = {
            leg.underlying_id for leg in option.legs
        }
        current_position = self.position.option_quantity_by_option_id.get(
            option.option_id, 0
        )

        other_gross = 0
        other_underlying_gross = defaultdict(int)

        for option_id, quantity in (
            self.position.option_quantity_by_option_id.items()
        ):
            if quantity == 0 or option_id == option.option_id:
                continue
            active = active_by_id.get(option_id)
            if active is None:
                continue

            other_gross += abs(quantity)
            active_underlyings = {
                leg.underlying_id for leg in active.legs
            }
            for underlying_id in active_underlyings:
                other_underlying_gross[underlying_id] += abs(quantity)

        capacity = 0
        maximum_test = min(requested_cap, 200)

        for quantity in range(1, maximum_test + 1):
            new_position = current_position + direction * quantity

            if abs(new_position) > per_option_limit:
                break
            if other_gross + abs(new_position) > gross_limit:
                break
            if any(
                other_underlying_gross[underlying_id]
                + abs(new_position)
                > underlying_limit
                for underlying_id in candidate_underlyings
            ):
                break

            capacity = quantity

        return capacity

    def _risk_reducing_quantity(
        self,
        option: BinaryOption,
        direction: int,
        requested_cap: int,
    ) -> int:
        current_position = self.position.option_quantity_by_option_id.get(
            option.option_id, 0
        )
        if direction < 0 and current_position > 0:
            return min(current_position, requested_cap)
        if direction > 0 and current_position < 0:
            return min(-current_position, requested_cap)
        return 0

    def _quantity_for_side(
        self,
        option: BinaryOption,
        direction: int,
        loss_per_contract: float,
        toxicity: float,
        portfolio_skew: float,
        fair_value: float,
    ) -> int:
        buffer = (
            self._dynamic_cash_buffer_fraction()
            * self.initial_cash_balance
        )
        free_cash = max(0.0, self.cash_balance - buffer)

        position_capacity = self._position_capacity(
            option, direction, self.MAX_QUOTE_SIZE
        )

        if loss_per_contract <= 1e-12:
            cash_capacity = self.MAX_QUOTE_SIZE
        else:
            cash_capacity = int(free_cash / loss_per_contract)

        # Normally place only a moderate fraction of capital at risk per fill.
        target_risk_budget = min(
            self.QUOTE_RISK_FRACTION * self.initial_cash_balance,
            0.30 * free_cash,
        )
        if loss_per_contract <= 1e-12:
            target_quantity = self.MAX_QUOTE_SIZE
        else:
            target_quantity = max(
                1, int(target_risk_budget / loss_per_contract)
            )

        # RFQ toxicity is informative but weaker than fully specified FOK flow.
        toxicity_scale = 1.0 / (1.0 + 8.0 * toxicity)

        # direction * portfolio_skew approximates the marginal change in
        # covariance risk.  Quote more size when a fill reduces economic risk
        # and less when it compounds an already concentrated exposure.
        directional_portfolio_risk = direction * portfolio_skew
        if directional_portfolio_risk > 0.0:
            inventory_scale = 1.0 / (
                1.0 + 10.0 * directional_portfolio_risk
            )
        else:
            inventory_scale = 1.0 + 4.0 * min(
                -directional_portfolio_risk, 0.12
            )

        scaled_target_quantity = max(
            1,
            int(
                target_quantity
                * toxicity_scale
                * inventory_scale
                * self._dynamic_risk_scale()
            ),
        )

        # Price can be competitive in the tails without carrying a large
        # directional bet.  Reduce only the side that adds statistically
        # dangerous exposure; never reduce an exact inventory-flattening side.
        current_inventory = (
            self.position.option_quantity_by_option_id.get(
                option.option_id, 0
            )
        )
        dangerous_low_probability_buy = (
            fair_value < 0.20
            and direction > 0
            and current_inventory >= 0
        )
        dangerous_high_probability_sale = (
            fair_value > 0.80
            and direction < 0
            and current_inventory <= 0
        )
        if (
            dangerous_low_probability_buy
            or dangerous_high_probability_sale
        ):
            scaled_target_quantity = max(
                1, int(0.50 * scaled_target_quantity)
            )

        # A trade that moves an existing position toward zero deserves enough
        # size to flatten it.  The separate cash_capacity check remains because
        # the challenge reserves collateral for every trade independently.
        flattening_quantity = self._risk_reducing_quantity(
            option, direction, self.MAX_QUOTE_SIZE
        )
        target_quantity = max(
            scaled_target_quantity, flattening_quantity
        )

        return max(
            0,
            min(
                self.MAX_QUOTE_SIZE,
                position_capacity,
                cash_capacity,
                target_quantity,
            ),
        )

    def _fok_required_edge(
        self,
        option: BinaryOption,
        fair_value: float,
    ) -> float:
        """Residual edge after informed flow has stressed the fair value.

        Size, toxicity, short tenor and repeated flow already enter the
        directional probability stress below.  Charging them again here would
        make V6 reject the same information twice.
        """
        model_error = (
            self.FOK_MODEL_ERROR_MULTIPLIER
            * self._model_standard_error(fair_value)
        )
        volatility_penalty = min(
            0.005,
            0.10 * self._recent_fair_volatility(option),
        )
        return self._clip(
            self.FOK_BASE_EDGE
            + model_error
            + volatility_penalty,
            self.FOK_BASE_EDGE,
            self.FOK_MAX_REQUIRED_EDGE,
        )

    def _fok_stressed_fair(
        self,
        option: BinaryOption,
        fok_order: FokOrder,
        fair_value: float,
        reservation_price: float,
    ) -> tuple[float, float, float, float]:
        """Stress fair value in the direction implied by an unusual FOK.

        A customer BUY is evidence for a higher payoff probability; a SELL is
        evidence for a lower probability.  Price surprise determines whether
        the signal is active.  Size, tenor, toxicity and repeated same-side
        requests determine its strength.  The logit representation preserves
        the probability bounds without special-case clipping near zero or one.
        """
        customer_direction = (
            1.0
            if fok_order.order_type == OrderType.BUY
            else -1.0
        )
        directional_price_surprise = max(
            customer_direction * (fok_order.price - fair_value),
            0.0,
        )
        uncertainty_scale = max(
            0.03,
            self._model_standard_error(fair_value)
            + 0.50 * self._recent_fair_volatility(option),
        )
        surprise_z = min(
            directional_price_surprise / uncertainty_scale,
            3.0,
        )
        surprise_score = surprise_z / 3.0

        size_score = self._clip(
            (fok_order.quantity - 2.0) / 24.0,
            0.0,
            1.0,
        )
        short_tenor_score = 1.0 / math.sqrt(
            max(option.steps_until_expiry, 1)
        )
        toxicity = self._counterparty_toxicity(
            fok_order.counterparty_id, option
        )
        toxicity_score = self._clip(toxicity / 0.05, 0.0, 1.0)

        request_key = (
            fok_order.counterparty_id,
            self._option_risk_key(option),
            int(customer_direction),
        )
        repeated_same_side = max(
            self.fok_directional_requests_this_step[request_key] - 1,
            0,
        )
        repeat_score = self._clip(
            repeated_same_side / 3.0, 0.0, 1.0
        )

        information_logit_shift = surprise_score * (
            0.65
            + 1.25 * size_score
            + 0.55 * short_tenor_score
            + 0.75 * toxicity_score
            + 0.25 * repeat_score
        )
        information_logit_shift = min(
            information_logit_shift,
            self.FOK_MAX_INFORMATION_LOGIT_SHIFT,
        )

        bounded_fair = self._clip(fair_value, 1e-6, 1.0 - 1e-6)
        fair_logit = math.log(bounded_fair / (1.0 - bounded_fair))
        stressed_fair = self._logistic(
            fair_logit
            + customer_direction * information_logit_shift
        )

        # Preserve the independently calculated inventory/flow adjustment.
        stressed_reservation = self._clip(
            reservation_price + stressed_fair - fair_value,
            0.0,
            1.0,
        )
        informed_probability_score = 1.0 - math.exp(
            -information_logit_shift
        )
        return (
            stressed_fair,
            stressed_reservation,
            informed_probability_score,
            information_logit_shift,
        )

    # ------------------------------------------------------------------
    # Required RFQ and FOK methods
    # ------------------------------------------------------------------

    def quote(  # type: ignore[empty-body]
        self,
        option: BinaryOption,
        counterparty_id: int,
    ) -> Quote:
        self.order_opportunities_this_step += 1
        self.counterparty_requests_this_step[counterparty_id] += 1
        (
            fair_value,
            reservation_price,
            half_spread,
            toxicity,
            portfolio_skew,
        ) = self._quote_components(option, counterparty_id)

        # Round outward, never inward, to the required penny grid.
        bid_price = round(
            max(
                0.0,
                min(
                    0.99,
                    math.floor(
                        (reservation_price - half_spread) * 100.0
                        + 1e-12
                    )
                    / 100.0,
                ),
            ),
            2,
        )
        offer_price = round(
            min(
                1.0,
                max(
                    0.01,
                    math.ceil(
                        (reservation_price + half_spread) * 100.0
                        - 1e-12
                    )
                    / 100.0,
                ),
            ),
            2,
        )

        if bid_price >= offer_price:
            bid_price = max(0.0, round(offer_price - 0.01, 2))

        bid_quantity = self._quantity_for_side(
            option,
            direction=1,
            loss_per_contract=bid_price,
            toxicity=toxicity,
            portfolio_skew=portfolio_skew,
            fair_value=fair_value,
        )
        offer_quantity = self._quantity_for_side(
            option,
            direction=-1,
            loss_per_contract=1.0 - offer_price,
            toxicity=toxicity,
            portfolio_skew=portfolio_skew,
            fair_value=fair_value,
        )

        # Quote requires positive quantities. If a side has no safe capacity,
        # move that side to a zero-worst-loss price and quote the minimum size.
        if bid_quantity == 0:
            bid_price = 0.0
            bid_quantity = 1
        if offer_quantity == 0:
            offer_price = 1.0
            offer_quantity = 1

        result = Quote(
            bid_price=bid_price,
            bid_quantity=bid_quantity,
            offer_price=offer_price,
            offer_quantity=offer_quantity,
        )

        self._debug(
            "quote",
            option_id=option.option_id,
            counterparty=counterparty_id,
            fair=round(fair_value, 6),
            reservation=round(reservation_price, 6),
            half_spread=round(half_spread, 6),
            toxicity=round(toxicity, 6),
            portfolio_skew=round(portfolio_skew, 6),
            bid=(result.bid_price, result.bid_quantity),
            offer=(result.offer_price, result.offer_quantity),
            cash=round(self.cash_balance, 4),
        )
        return result

    def respond_to_fok(  # type: ignore[empty-body]
        self,
        option: BinaryOption,
        fok_order: FokOrder,
    ) -> bool:
        self.order_opportunities_this_step += 1
        self.counterparty_requests_this_step[
            fok_order.counterparty_id
        ] += 1
        customer_direction = (
            1
            if fok_order.order_type == OrderType.BUY
            else -1
        )
        directional_request_key = (
            fok_order.counterparty_id,
            self._option_risk_key(option),
            customer_direction,
        )
        self.fok_directional_requests_this_step[
            directional_request_key
        ] += 1

        (
            fair_value,
            reservation_price,
            _half_spread,
            _toxicity,
            _portfolio_skew,
        ) = self._quote_components(option, fok_order.counterparty_id)

        (
            stressed_fair,
            stressed_reservation,
            informed_probability_score,
            information_logit_shift,
        ) = self._fok_stressed_fair(
            option,
            fok_order,
            fair_value,
            reservation_price,
        )

        buffer = (
            self._dynamic_cash_buffer_fraction()
            * self.initial_cash_balance
        )

        required_edge = self._fok_required_edge(
            option,
            fair_value,
        )
        payoff_variance = fair_value * (1.0 - fair_value)
        concentration_charge = min(
            0.08,
            0.50
            * self._dynamic_risk_aversion()
            * fok_order.quantity
            * payoff_variance,
        )

        if fok_order.order_type == OrderType.BUY:
            # Customer buys; this market maker sells.
            direction = -1
            stressed_edge = fok_order.price - stressed_reservation
            maximum_loss = fok_order.quantity * max(
                1.0 - fok_order.price, 0.0
            )
            position_capacity = self._position_capacity(
                option,
                direction=direction,
                requested_cap=fok_order.quantity,
            )
        else:
            # Customer sells; this market maker buys.
            direction = 1
            stressed_edge = stressed_reservation - fok_order.price
            maximum_loss = fok_order.quantity * max(fok_order.price, 0.0)
            position_capacity = self._position_capacity(
                option,
                direction=direction,
                requested_cap=fok_order.quantity,
            )

        attractive = (
            stressed_edge >= required_edge + concentration_charge
        )
        affordable = self.cash_balance - maximum_loss >= buffer
        within_limits = position_capacity >= fok_order.quantity

        fully_risk_reducing = (
            self._risk_reducing_quantity(
                option, direction, fok_order.quantity
            )
            >= fok_order.quantity
        )
        loss_fraction = (
            0.25 if fully_risk_reducing else self.MAX_FOK_LOSS_FRACTION
        )
        loss_budget = min(
            max(self.cash_balance - buffer, 0.0),
            loss_fraction * self.initial_cash_balance,
        )
        within_loss_budget = maximum_loss <= loss_budget + 1e-12

        accepted = (
            attractive
            and affordable
            and within_limits
            and within_loss_budget
        )

        self._debug(
            "fok",
            option_id=option.option_id,
            counterparty=fok_order.counterparty_id,
            side=str(fok_order.order_type),
            price=fok_order.price,
            quantity=fok_order.quantity,
            fair=round(fair_value, 6),
            reservation=round(reservation_price, 6),
            stressed_fair=round(stressed_fair, 6),
            stressed_reservation=round(stressed_reservation, 6),
            stressed_edge=round(stressed_edge, 6),
            informed_score=round(informed_probability_score, 6),
            information_logit_shift=round(
                information_logit_shift, 6
            ),
            required_edge=round(required_edge, 6),
            concentration_charge=round(concentration_charge, 6),
            maximum_loss=round(maximum_loss, 4),
            loss_budget=round(loss_budget, 4),
            attractive=attractive,
            affordable=affordable,
            within_limits=within_limits,
            accepted=accepted,
        )

        return accepted

    # ------------------------------------------------------------------
    # Rolling/EWMA market-parameter estimation
    # ------------------------------------------------------------------

    def _estimate_parameters_from_history(self) -> None:
        rates = list(self.rate_history)
        ajr_values = list(self.ajr_history)
        thr_values = list(self.thr_history)

        number_of_steps = len(rates) - 1
        self.history_steps = max(number_of_steps, 1)

        if number_of_steps <= 0:
            self.estimated_parameters = MarketParameters(
                ajarai_drift=0.0,
                ajarai_idio_std_dev=0.05,
                ajarai_rate_beta=0.0,
                ajarai_sector_beta=0.03,
                rate_down_probability=0.25,
                rate_reversion_strength=0.05,
                rate_up_probability=0.25,
                sector_std_dev=1.0,
                theriodic_drift=0.0,
                theriodic_idio_std_dev=0.05,
                theriodic_rate_beta=0.0,
                theriodic_sector_beta=0.03,
            )
            return

        # Exponential weighting keeps all available observations in the rolling
        # window while emphasizing recent days. With little burn-in, the
        # half-life automatically shortens rather than assuming 60/200 days.
        half_life = min(
            self.MARKET_HALF_LIFE,
            max(5.0, number_of_steps / 2.0),
        )
        decay = math.exp(-math.log(2.0) / half_life)
        step_weights = [
            decay ** (number_of_steps - 1 - index)
            for index in range(number_of_steps)
        ]
        total_step_weight = sum(step_weights)

        rate_changes = [
            round(rates[index + 1] - rates[index], 2)
            for index in range(number_of_steps)
        ]

        # At rate zero, a downward draw is observed as a flat move, so exclude
        # that censored state when fitting the ordinary up/down regression.
        usable_rate_indices = [
            index
            for index in range(number_of_steps)
            if rates[index] > 1e-12
        ]
        if not usable_rate_indices:
            usable_rate_indices = list(range(number_of_steps))

        usable_rate_weights = [
            step_weights[index] for index in usable_rate_indices
        ]
        total_rate_weight = sum(usable_rate_weights)

        distance_from_target = [
            2.0 - rates[index] for index in usable_rate_indices
        ]
        up_indicators = [
            float(rates[index + 1] > rates[index])
            for index in usable_rate_indices
        ]
        down_indicators = [
            float(rates[index + 1] < rates[index])
            for index in usable_rate_indices
        ]

        x_mean = sum(
            weight * value
            for weight, value in zip(
                usable_rate_weights, distance_from_target
            )
        ) / total_rate_weight
        up_mean = sum(
            weight * value
            for weight, value in zip(
                usable_rate_weights, up_indicators
            )
        ) / total_rate_weight
        down_mean = sum(
            weight * value
            for weight, value in zip(
                usable_rate_weights, down_indicators
            )
        ) / total_rate_weight
        x_sum_of_squares = sum(
            weight * (value - x_mean) ** 2
            for weight, value in zip(
                usable_rate_weights, distance_from_target
            )
        )

        if x_sum_of_squares > 1e-12:
            numerator = sum(
                weight
                * (x_value - x_mean)
                * ((up - up_mean) - (down - down_mean))
                for weight, x_value, up, down in zip(
                    usable_rate_weights,
                    distance_from_target,
                    up_indicators,
                    down_indicators,
                )
            )
            reversion_strength = self._clip(
                numerator / (2.0 * x_sum_of_squares),
                0.0,
                1.0,
            )
        else:
            reversion_strength = 0.05

        up_probability = max(
            1e-4, up_mean - reversion_strength * x_mean
        )
        down_probability = max(
            1e-4, down_mean + reversion_strength * x_mean
        )

        if up_probability + down_probability > 0.999:
            probability_scale = 0.999 / (
                up_probability + down_probability
            )
            up_probability *= probability_scale
            down_probability *= probability_scale

        ajr_log_returns = [
            math.log(
                max(ajr_values[index + 1], 1e-12)
                / max(ajr_values[index], 1e-12)
            )
            for index in range(number_of_steps)
        ]
        thr_log_returns = [
            math.log(
                max(thr_values[index + 1], 1e-12)
                / max(thr_values[index], 1e-12)
            )
            for index in range(number_of_steps)
        ]

        rate_change_mean = sum(
            weight * change
            for weight, change in zip(step_weights, rate_changes)
        ) / total_step_weight
        rate_change_sum_of_squares = sum(
            weight * (change - rate_change_mean) ** 2
            for weight, change in zip(step_weights, rate_changes)
        )

        def fit_company(log_returns: list[float]):
            return_mean = sum(
                weight * log_return
                for weight, log_return in zip(step_weights, log_returns)
            ) / total_step_weight

            if rate_change_sum_of_squares > 1e-12:
                rate_beta = sum(
                    weight
                    * (change - rate_change_mean)
                    * (log_return - return_mean)
                    for weight, change, log_return in zip(
                        step_weights, rate_changes, log_returns
                    )
                ) / rate_change_sum_of_squares
            else:
                rate_beta = 0.0

            drift = return_mean - rate_beta * rate_change_mean
            residuals = [
                log_return - drift - rate_beta * change
                for change, log_return in zip(
                    rate_changes, log_returns
                )
            ]
            return drift, rate_beta, residuals

        ajr_drift, ajr_rate_beta, ajr_residuals = fit_company(
            ajr_log_returns
        )
        thr_drift, thr_rate_beta, thr_residuals = fit_company(
            thr_log_returns
        )

        ajr_residual_variance = max(
            sum(
                weight * residual ** 2
                for weight, residual in zip(
                    step_weights, ajr_residuals
                )
            )
            / total_step_weight,
            1e-12,
        )
        thr_residual_variance = max(
            sum(
                weight * residual ** 2
                for weight, residual in zip(
                    step_weights, thr_residuals
                )
            )
            / total_step_weight,
            1e-12,
        )
        residual_covariance = sum(
            weight * ajr_residual * thr_residual
            for weight, ajr_residual, thr_residual in zip(
                step_weights, ajr_residuals, thr_residuals
            )
        ) / total_step_weight

        # Short histories are noisy. Shrink estimates toward conservative
        # defaults, with data weight increasing smoothly as observations grow.
        data_weight = number_of_steps / (
            number_of_steps + self.PRIOR_SAMPLE_SIZE
        )
        up_probability = (
            data_weight * up_probability
            + (1.0 - data_weight) * 0.25
        )
        down_probability = (
            data_weight * down_probability
            + (1.0 - data_weight) * 0.25
        )
        reversion_strength = (
            data_weight * reversion_strength
            + (1.0 - data_weight) * 0.05
        )
        ajr_residual_variance = (
            data_weight * ajr_residual_variance
            + (1.0 - data_weight) * 0.05 ** 2
        )
        thr_residual_variance = (
            data_weight * thr_residual_variance
            + (1.0 - data_weight) * 0.05 ** 2
        )
        residual_covariance *= data_weight

        covariance_bound = math.sqrt(
            ajr_residual_variance * thr_residual_variance
        )
        residual_covariance = self._clip(
            residual_covariance,
            -covariance_bound,
            covariance_bound,
        )

        # Factor decomposition reproducing both marginal variances and covariance.
        # sector_std_dev is normalized to one because only beta*sector_std is
        # identifiable from these two return histories.
        if abs(residual_covariance) <= 1e-14:
            ajr_sector_loading = 0.0
            thr_sector_loading = 0.0
        else:
            variance_ratio = (
                ajr_residual_variance / thr_residual_variance
            ) ** 0.25
            ajr_sector_loading = (
                math.sqrt(abs(residual_covariance)) * variance_ratio
            )
            thr_sector_loading = math.copysign(
                math.sqrt(abs(residual_covariance)) / variance_ratio,
                residual_covariance,
            )

        ajr_idio_std = math.sqrt(
            max(
                ajr_residual_variance - ajr_sector_loading ** 2,
                0.0,
            )
        )
        thr_idio_std = math.sqrt(
            max(
                thr_residual_variance - thr_sector_loading ** 2,
                0.0,
            )
        )

        self.estimated_parameters = MarketParameters(
            ajarai_drift=data_weight * ajr_drift,
            ajarai_idio_std_dev=ajr_idio_std,
            ajarai_rate_beta=data_weight * ajr_rate_beta,
            ajarai_sector_beta=ajr_sector_loading,
            rate_down_probability=down_probability,
            rate_reversion_strength=reversion_strength,
            rate_up_probability=up_probability,
            sector_std_dev=1.0,
            theriodic_drift=data_weight * thr_drift,
            theriodic_idio_std_dev=thr_idio_std,
            theriodic_rate_beta=data_weight * thr_rate_beta,
            theriodic_sector_beta=thr_sector_loading,
        )

        self._clear_market_state_caches()

    # ------------------------------------------------------------------
    # Required warm-up entry point
    # ------------------------------------------------------------------

    def warm_up(  # type: ignore[empty-body]
        self,
        market_history: MarketHistory,
    ) -> None:
        all_rates = list(
            market_history.values_by_underlying_id[
                FED_FUNDS_RATE_UNDERLYING_ID
            ]
        )
        all_ajr_values = list(
            market_history.values_by_underlying_id[AJARAI_UNDERLYING_ID]
        )
        all_thr_values = list(
            market_history.values_by_underlying_id[THERIODIC_UNDERLYING_ID]
        )

        # Use only what is actually available. MARKET_WINDOW is a maximum, not
        # an assumption that 60 historical days exist before execution.
        available_days = len(all_rates)
        window = min(self.MARKET_WINDOW, available_days)

        self.rate_history = all_rates[-window:]
        self.ajr_history = all_ajr_values[-window:]
        self.thr_history = all_thr_values[-window:]
        self._estimate_parameters_from_history()

        self._debug(
            "warm_up",
            available_days=available_days,
            window=window,
            parameters=repr(self.estimated_parameters),
        )
