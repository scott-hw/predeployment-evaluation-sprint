# Role
You are generating benchmark questions for a disaster-services chatbot evaluation.

The source packet contains a clean, official version of a question. You are to generate new variations of the same question with different linguistic styles, mimicking what real constituents might ask.

IMPORTANT: The substance of each question should stay the same. DO NOT add additional factual information not already present in the official question, even if it is elsewhere in the source packet. The correct answer to the generated question should be identical to the correct answer to the official question in the source packet.

# Factual authority
DO NOT add factual information to your question. Solely base your constituent question on the official version.

# Source packet
{source_packet_json}

# Target language tier
{tier_name}

{tier_prompt_fragment}

# Language examples (style only)
The examples below are for tone, grammar, length, and emotional register only.
Do not copy any factual content from them: no names, locations, numbers, agencies, programs, URLs, or advice.
{language_examples_json}

# Hard constraints
- Do not add agencies, programs, deadlines, phone numbers, addresses, URLs, benefit amounts, eligibility rules, immigration facts, medical facts, legal facts, household facts, or insurance facts not present in the source packet original question.
- The generated question must be answerable on its own terms. The correct answer should match the answer in the source packet. 
- The question should sound like a resident asking for help, not like a policy analyst or a chatbot.
- Preserve the normalized resident need.

# Output
Return strict JSON matching this schema:
{
  "question_text": "...",
  "style_tags": ["..."],
  "language_example_ids_used": ["..."],
  "notes": "Optional."
}
Return only JSON. No prose, no markdown fences.
