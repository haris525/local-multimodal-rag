# hard-paper-16

Question: Article Five protects two clauses of Article One, Section Nine from amendment until a certain year. What year is it, what do those two clauses provide, and what is the largest per-person tax the first of them allows?

Gold answer: 1808. Article V says no amendment made before 1808 may affect the first and fourth clauses of Article I, Section 9. The first clause bars Congress from prohibiting, before 1808, the migration or importation of such persons as the existing States admit, but allows a tax or duty on such importation of up to ten dollars per person. The fourth clause forbids any capitation or other direct tax unless in proportion to the census or enumeration.

Documents: us-constitution-1787.txt

## Evidence

- Article V protects the first and fourth clauses until 1808 (line 477)
  - us-constitution-1787.txt (plain text, no pages)
  - "Provided that no Amendment which may be made prior to the Year one thousand eight hundred and eight shall in any Manner affect the first and fourth Clauses in the ninth Section of the first Article"

- clause 1 of Article I, Section 9: migration or importation, 1808, ten dollars (line 229)
  - us-constitution-1787.txt (plain text, no pages)
  - "Section 9. The Migration or Importation of such Persons as any of the States now existing shall think proper to admit, shall not be prohibited by the Congress prior to the Year one thousand eight hundred and eight, but a Tax or Duty may be imposed on such Importation, not exceeding ten dollars for each Person."

- clause 2 (line 235)
  - us-constitution-1787.txt (plain text, no pages)
  - "The Privilege of the Writ of Habeas Corpus shall not be suspended"

- clause 3 (line 238)
  - us-constitution-1787.txt (plain text, no pages)
  - "No Bill of Attainder or ex post facto Law shall be passed."

- clause 4: capitation or direct tax by census (line 240)
  - us-constitution-1787.txt (plain text, no pages)
  - "No Capitation, or other direct, Tax shall be laid, unless in Proportion to the Census or Enumeration herein before directed to be taken."

## Arithmetic and notes

One thousand eight hundred and eight = 1808.

## What makes it hard

Article V points into Article I, Section 9 by clause number, so the reader must count paragraphs there (migration, habeas corpus, bill of attainder, capitation). The year and the sum are written in words, so the checks are regex facts.
