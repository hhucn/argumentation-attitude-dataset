# Argumentation dataset with individual attitudes

## Definitions

Following the IBIS model, we have
* 2 *positions*:
    * Should plastic packaging for fresh food such as fruit and vegetables be allowed (0) or prohibited (1) in Germany? (id 363)
    * Should the growing of genetically modified plants for food production be allowed (0) or prohibited (1) in Germany? (id 324)
* *arguments* for/against each position

Each participant indicated
* their *opinion on the positions* (0/1)
* whether they consider an argument convincing (1) or not (0)
* their *opinion strength* (0-6)

## Overview

A certain set of arguments have been provided by us before, more arguments have been added by participants.

* plastic packaging: 36+521 arguments
* genetic engineering: 38+351 arguments

## Data collection

The data has been collected at four different points of time, where different participants provided their attitudes on positions and arguments formulated by us:

* T0: Pre-test data with 264 participants; opinions and opinion strengths on the positions plastic packaging and genetic engineering; opinions and convincingness on 14 randomly selected arguments per topic
* T1: first main experiment with 410 participants; opinions and opinion strengths on plastic packaging and genetic engineering
* T2: second main experiment with 289 participants (subset of users from T1); opinions and opinion strengths on plastic packaging and genetic engineering; opinions and convincingness on 6 randomly selected argument for/against plastic packing (3 randomly selected supporting, and 3 randomly selecting attacking arguments); users were able to contribute own arguments on the topic plastic packaging (relevant statement ids: 364-399)
* T3: third main experiment with 229 participants (subset of users from T2); opinions and opinion strengths on plastic packaging and genetic engineering; opinions and convincingness on 6 randomly selected argument for/against genetic engineering (3 randomly selected supporting, and 3 randomly selecting attacking arguments); users were able to contribute own arguments on the topic genetic engineering (relevant statement ids: 325-362)

For details on the data collection and the context of the original experiment (which included more groups and users, where the presentation of arguments was different), see [Kelm et al.](https://www.sciencedirect.com/science/article/pii/S2451958823000763?via%3Dihub).

### Exact questions

The subjective assessment of the test subjects was recorded. The subjects themselves decided whether an argument was pro/contra and how strong it was considered to be. The exact wording of the questions was:

- Die folgenden Argumente hat ein Algorithmus aus der Menge von Argumenten ausgewählt, die andere Personen genannt haben. Stimme Sie diesen Argumenten eher zu oder eher nicht zu? stimme zu / stimme nicht zu
  - The following arguments have been selected by an algorithm from the set of arguments that other people have mentioned. Do you tend to agree or disagree with these arguments? agree / disagree
- Geben Sie an, wie stark Sie die Argumente zu `<Position>` finden. sehr schwach - - - - - sehr stark
  - Indicate how strongly you agree with the arguments about `<position>`. very weak - - - - - - very strong
- Stimmen Sie dieser Aussage eher zu oder eher nicht zu? `<Position>` Ich stimme zu / Ich stimme nicht zu
  - Do you agree or disagree with this statement? `<position>` I agree / I disagree
- Wie sicher sind Sie sich mit Ihrer Meinung? sehr unsicher - - - - - - - sehr sicher
  - How sure are you of your opinion? very unsure - - - - - - - - very sure

## Provided files

We provide the following files:

* `arguments.csv`: all statements and positions, as provided by us or contributed by the participants:
    * `statement_id`: ID of the statement
    * `conclusions`: list of statement sid which were used as conclusion for this statement (empty for positions) and the information whether the formed argument is supportive (`+`) or attacking (`-`)
    * `text`: original (German) text of this statement
    * `text_en`: English translation of the text
    * `author`: the author of the argument, either UPEKI (we), or a user name
* `arguments.json`: all arguments in the Argument Interchange Format (AIF)
* `train.csv`
    * for each user, contains the agreement(1)/disagreement(0) attitude information for positions/arguments, as well as strength ratings
    * `rating_after` is the value for a position provided at that point of time (T1 or T2)
    * `rating_before` is the value for a position provided at the previoud point of time (T2 or T3)
    * folder `T1_T2`: data contains information after T1, before T2
        * i.e. complete data for pre-test participants, only opinion on positions for main experiment participants
    * folder `T2_T3`: data contains information after T2, before T3
        * i.e. complete data for pre-test participants, argument attitudes for plastic packaging for main experiment participants
* `validation.csv`
    * same structure as `train.csv`
    * folder `T1_T2`: data contains information after T2, before T3
        * i.e. argument attitudes for plastic packaging for half of the main experiment participants
    * folder `T1_T3`: data contains information after T3
        * i.e. complete data for half of the main experiment participants
* `test.csv`
    * same as `validation.csv`, but for the other half of main experiment participants
* `T0.csv`, `T1.csv`, `T2.csv`, `T3.csv` in `complete`:
    * same structure as `train.csv` for the individual T1→T2/T2→T3 sets
    * contains the complete data from the points of time, without any split


## Relevant publications

Markus Brenneis, Maike Behrendt, and Stefan Harmeling (July 2021). “How Will I Argue? A Dataset for Evaluating Recommender Systems for Argumentations”. In: *Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue.* Singapore and
Online: Association for Computational Linguistics, pp. 360–367
