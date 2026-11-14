# hard-paper-15

Question: Using only the Constitution's text: how is the number of presidential electors for a State determined, what is the smallest number of electors any State can have, and how many electors would Virginia have had under the House seats the Constitution assigns before the first census?

Gold answer: Article II, Section 1: each State appoints electors equal to the whole number of its Senators and Representatives. Article I, Section 3 gives every State two Senators and Article I, Section 2 guarantees each State at least one Representative, so the minimum is 3 electors. Before the first enumeration, Article I, Section 2 assigns Virginia ten Representatives, so Virginia would have had 10 + 2 = 12 electors.

Documents: us-constitution-1787.txt

## Evidence

- electors = Senators + Representatives (Article II, Section 1; line 287)
  - us-constitution-1787.txt (plain text, no pages)
  - "Each State shall appoint, in such Manner as the Legislature thereof may direct, a Number of Electors, equal to the whole Number of Senators and Representatives to which the State may be entitled in the Congress"

- two Senators per State (Article I, Section 3; lines 51-52)
  - us-constitution-1787.txt (plain text, no pages)
  - "The Senate of the United States shall be composed of two Senators from each State"

- at least one Representative (Article I, Section 2; line 38)
  - us-constitution-1787.txt (plain text, no pages)
  - "but each State shall have at least one Representative;"

- seats before the first enumeration (Article I, Section 2; line 39)
  - us-constitution-1787.txt (plain text, no pages)
  - "and until such enumeration shall be made, the State of New Hampshire shall be entitled to chuse three"

- Virginia ten (Article I, Section 2; line 43)
  - us-constitution-1787.txt (plain text, no pages)
  - "Virginia ten, North Carolina five, South Carolina five, and Georgia three."

## Arithmetic and notes

Minimum: 2 Senators + 1 Representative = 3. Virginia: 2 Senators + 10 Representatives = 12. (Delaware and Rhode Island, with one seat each, would have had the minimum of 3.)

## What makes it hard

Three clauses in two articles: Article II, Section 1 (electors), Article I, Section 3 (two Senators) and Article I, Section 2 (at least one Representative; Virginia's ten seats). Numbers are written as words in the text, so the checks are regex facts.
