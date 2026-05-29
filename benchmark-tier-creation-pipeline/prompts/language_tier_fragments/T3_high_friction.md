# Language tier: T3 — High Friction

Write a fragmented, SMS-shorthand style question. This is the hardest tier for NLP parsing.

Requirements:
- May omit articles (a, an, the) and auxiliary verbs
- Lowercase throughout is acceptable
- Use abbreviations: fema, ca, pls, asap, idk, govt, w/, b/c, info
- Fragmented phrasing — partial sentences are fine ("lost home fire need help rent")
- May include emotional register: urgent, distressed, confused
- Maximum two short fragments or one run-on phrase
- Intent must still be inferable (even if NLP parsing is hard)
- Both underusing and overusing punctuation are acceptable options

Style tags to use: ["T3", "fragmented", "sms_shorthand", "high_friction", "lowercase"]
