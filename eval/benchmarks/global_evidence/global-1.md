# global-1

Question: Using only the text of the original 1787 Constitution: which federal offices does it set a minimum age for? For each, give the minimum age, the required years of citizenship or residence, and any other personal qualification the text states for that office.

Gold answer: Three offices. A Representative (Article I, Section 2) must be at least 25 years old, a citizen of the United States for seven years, and, when elected, an inhabitant of the State in which he is chosen. A Senator (Article I, Section 3) must be at least 30, a citizen for nine years, and, when elected, an inhabitant of the State for which he is chosen. The President (Article II, Section 1) must be a natural born citizen (or a citizen at the time the Constitution was adopted), at least 35 years old, and fourteen years a resident within the United States. The original text sets no separate age for the Vice President.

Documents: us-constitution-1787.txt

## Evidence

- Representative: 25, seven years a citizen, inhabitant of the State (Article I, Section 2; lines 23-26)
  - us-constitution-1787.txt (plain text, no pages)
  - "No Person shall be a Representative who shall not have attained to the Age of twenty five Years, and been seven Years a citizen of the United States, and who shall not, when elected, be an Inhabitant of that State in which he shall be chosen."

- Senator: 30, nine years a citizen, inhabitant of the State (Article I, Section 3; lines 65-68)
  - us-constitution-1787.txt (plain text, no pages)
  - "No person shall be a Senator who shall not have attained to the Age of thirty Years, and been nine Years a Citizen of the United States, and who shall not, when elected, be an Inhabitant of that State for which he shall be chosen."

- President: natural born citizen, 35, fourteen years a resident (Article II, Section 1; lines 322-326)
  - us-constitution-1787.txt (plain text, no pages)
  - "No Person except a natural born Citizen, or a Citizen of the United States, at the time of the Adoption of this Constitution, shall be eligible to the Office of President; neither shall any Person be eligible to that Office who shall not have attained to the Age of thirty five Years, and been fourteen Years a Resident within the United States."

## Arithmetic and notes

The ages and years are written as words in the text, so the checks are regex facts that accept digits or words. Completeness: "Age of" occurs exactly three times (lines 24, 65, 325) and "Years a" exactly three times (lines 24, 66, 326). The only other "Inhabitant" (line 294) is about electors voting for two persons, not a qualification for an office. The file holds only the 1787 text (no amendments), so there is no Vice President qualification clause (that came with the Twelfth Amendment).

## What makes it global

Article I, Section 2, Article I, Section 3 and Article II, Section 1 each hold one third of the answer, about 300 lines apart; an answer built from the Congress passages alone misses the President, and one built from the President passage misses the inhabitant rule for Congress.
