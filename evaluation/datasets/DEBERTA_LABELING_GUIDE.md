# Labelling guide: DeBERTa sample

Rules for filling `data/raw/deberta_sheet.md`. The sample measures `neutral`'s DeBERTa, which was
trained on BABE, so "biased" here has to mean what it means in BABE. This page summarises BABE's
annotation manual ("Annotation Manual Expert Dataset", Media Bias Project, 6 November 2020,
`annotation_guidelines_BABE.pdf` in the BABE repository). Every example below is invented for this
guide; none comes from BABE or from an outlet.

## The three labels

- `b` biased: the sentence's own wording conveys a judgement.
- `n` neutral: it does not. BABE calls this "non-biased".
- `s` skip: not a sentence that can be judged, such as a photo caption fragment, a list of names,
  or a line cut off by the sentence splitter. Skipped sentences are left out of the score.

## What counts as biased

BABE looks only at language. A sentence is biased when the way it is written pushes the reader
toward a view, in one of two ways.

**Framing: labelling and word choice.**

- An adjective or adverb that carries a strong opinion about a person, place or event.
  "The commission announced the *long-overdue* recount" is biased; "The commission announced the
  recount" is not.
- A one-sided or loaded term where a plain one exists. "The *regime's* candidate" against "the
  government-backed candidate"; "protesters *swarmed* the office" against "protesters gathered at
  the office".

**Epistemological: wording that makes a statement more or less believable.**

- A verb that presupposes the statement is true or false. "The minister *admitted* the budget was
  cut" presents the cut as a fact he was reluctant to concede; "the minister *said* the budget
  was cut" only reports his words. "The party
  *claimed* the count was fair" invites doubt; "the party *said*" does not.
- Stating as fact what is one side's view, or hedging what is established. "He *realised* the
  ruling was political" against "he *said he believed* the ruling was political".

"Say" and "state" are neutral. "Claim", "boast", "admit", "blast" and similar verbs are the usual
signs of this second kind.

## What does not count

1. **Truth is not the question.** A sentence can be wrong and still neutral, or correct and still
   biased. Do not check facts.
2. **A controversial topic is not bias.** "The court heard the vote-buying case" names a
   contested subject in plain words: neutral.
3. **Quoted or reported speech does not excuse the wording.** Judge the words printed in the
   sentence, whoever said them. `The rival called the ruling "a disgrace"` is biased, and so is
   "the count was riddled with fraud, the campaign team said": the loaded words are in the
   sentence the reader sees. "The rival said he disagreed with the ruling" is neutral. BABE's
   manual has no rule for direct quotes, but its example (2) counts a reported statement
   ("massive wave of migrants that's poured across the border ..., beleaguered border officials
   said") as biased, and the model reads only the words. The owner confirmed this rule on 2026-11-15;
   the other reading, where only the outlet's own words count, would give fewer `b`.
4. **Your own politics stay out.** Label biased language whichever side it favours.

## How to label

1. Read the sentence alone. The sheet shows no outlet, no article and no model output on purpose.
2. Ask: is there a word or short phrase I could underline as carrying the judgement? If yes, `b`.
   If the sentence would read the same after swapping in plain words, because it already uses
   them, `n`.
3. When unsure between `b` and `n`, pick the closer one. `s` is for text that is not a judgeable
   sentence, not for hard cases.
4. One pass, in order, without going back to even out the counts. No share of `b` is expected
   or wanted.

## Limits of this guide

BABE's annotators also highlighted the biased words and labelled each sentence factual or
opinionated; this sample collects only the biased / neutral label. The owner is one annotator
working from this summary, not a trained BABE annotator, and the report says so.
