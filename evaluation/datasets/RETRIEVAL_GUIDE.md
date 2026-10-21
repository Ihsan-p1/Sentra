# Labelling guide: retrieval gold

Rules for the gold sheet written by `scripts/make_retrieval_pool.py sheet`. For each question,
the owner lists the candidate articles that are relevant. Recall@5 and MRR@10 are computed
against that list, so the rules decide what the numbers mean.

## The two labels

- **Relevant**: write the candidate's number on the `relevan:` line.
- **Not relevant**: leave the number out. Write `-` when no candidate is relevant.

## Rules

1. **Only the sheet counts.** Judge from the title and the text shown, not from what you know
   about the news or from the URL.
2. **Relevant means it helps answer.** An article is relevant when it states at least one
   fact that a good answer to the question would use. Answering part of the question is
   enough.
3. **Mentioning the topic is not enough.** An article that names the topic but says nothing
   that answers the question is not relevant. For "who won the 2024 election", an article
   about Bawaslu supervising overseas voting is not relevant; one reporting the quick count
   is.
4. **A passing mention counts only if it answers.** "President-elect Prabowo" in an article
   about something else answers "who won" in part, so it is relevant. "Prabowo" named in
   passing does not answer "what is Prabowo's foreign policy".
5. **Scope in the question is binding.** If the question names a year, an event or an
   outlet, the article must match it. An article about 2019 does not answer a question about
   2024.
6. **Comparison questions.** For "how did the outlets frame X", any article from any outlet
   that covers X is relevant, because each outlet's coverage is part of the answer.
7. **Accuracy is not judged.** Mark whether the article addresses the question, not whether
   it is right. Two articles that contradict each other can both be relevant.
8. **Judge each article on its own.** The same story from two outlets is relevant twice.
9. **Indonesian questions follow the same rules.** "siapa itu prabowo subianto" is judged
   exactly like "who is prabowo subianto".
10. **When unsure, leave it out.** Only articles you would point to as evidence are relevant.

## When a rule changes

Add a line below with the date and the rule number, and rejudge the questions that rule
affects. Otherwise questions labelled before and after the change are measured against
different standards.

- 2026-10-21: guide drafted by Claude and confirmed by the owner, before the first label.
