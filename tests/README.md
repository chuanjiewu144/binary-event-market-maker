# Planned validation suite

Automated tests have not yet been implemented.

Planned coverage includes:

1. fair values remain within `[0, 1]`;
2. RFQ prices lie on the penny grid;
3. every quote satisfies `bid < offer`;
4. quote quantities are positive;
5. customer BUY and SELL signs are mapped correctly;
6. accepted FOK orders preserve the cash buffer;
7. long sequences of one-sided orders do not cause bankruptcy;
8. extreme-probability options receive asymmetric quantity protection;
9. deterministic seeds reproduce identical valuations; and
10. previously observed large-order failures remain rejected.

A `pytest` step will be added to continuous integration after the first
real tests are implemented.