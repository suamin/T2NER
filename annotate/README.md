## T2NER-annotate: A light-weight Python annotation tool for NER
This directory contains a simple, quick-to-use and light-weight annotation tool for named entity recognition (NER). The tool expects unannotated input data in the CoNLL format.

#### Instructions
- Run the help menu with `python annotate.py --help` and follow the arguments.
- The script will create a working directory per source annotation file. This directory will contain all the annotator progress and backup of annotations.
- By default all sentence tokens are marked as **O** (Outside) tag from the start.
- During annotation, main keys are:
   - `d` : Once a sentence has been annotated, annotator can press `d` to indicate **done**. To make sure, the tool will display the current sentence and its NER tags and ask for `Are you sure? (y/n)`. Press `y` to indicate yes, any other key will bring back the annotating sentence.
    - `s` : Indicates that you wish to **skip** this sentence. The tool keeps track of the difference between a skipped and annotated sentence. Rule of thumb to use this command is when annotator is unsure about the annotation. A second usecase would be when sentences are too noisy or short for example.
    - `h` : Press `h` for **hint** anytime during annotation to request the English translation (if available) for a sentence, if available. This option is only provided to support the annotator if the sentence is quite difficult to understand / ambigious / code-mixed with other languages.
    - `c` : You can **clear** the current annotation. This resets the tag of all the tokens in this sentence back to **O**. This command can be used when to reset the labels, if annotator feels many mistakes were made. Clearing a specific position is not supported at the moment, instead simply re-annotate the desired positions.

#### Note
It is solely annotators responsibility to provide the clean tags to the best of their knowledge. The tool is highly focused towards fast light-weight annotations and not industrial grade to support all the features along with intricate error checking.

#### Tips
To minimize overwriting your existing sessions, create new working directory everytime if you want to start from scratch. 
