# PyTorch Transformer Studies

The following repository contains:
 * A working Transformer implementation that uses `nn.Transformer` for language translation
 * Papers I've analyzed during my NLP research in separate Jupyter Notebooks
 * "Test" notebook that displays the mini-framework in action

Originally, I planned to analyze 5 papers for the Transformer architecture, but found 2 to be satisfactory. I'm not sure if this applies to everyone as I was already familiar with other NLP approaches and some techniques used in the Transformer before. For each paper, I followed these steps to make sure I grokked how Transformers work:
 * Identify main ideas
 * Try to replicate in PyTorch
 * Find alternatives online
 * Improve my solution and iterate
 * Hypothesize causes of inadequate results
 
This also stands as a documentation of obscure `nn.Transformer` modules. Some of their functions or passed parameters aren't explained well in the documentation. For example, I've identified the following to be unintuitive:
 * Feeding data - how do Transformers allow parallelization
 * Feeding data - when do we cut off `<eos>` and `<sos>` tokens
 * Data transformations and their dimensions during feed-forward
 * Padding masks, attention masks and in which layers they are applied
 
Consequently, I've documented these issues where they arise.
