# WARNING!

This code is newly baked.. so you have been warned..  I will be doing a meetup so thought I'd publish before I'm completely happy
with this code.

# Deep Spelling Tree

This is an experimental project where you can use deep learning to do spell checking on words.  

Below is a loose description of how the algorithm works.

My is simple we encode words with a custom encoding and then create synthetic word versions with the same categorization. The synthetic words represent possible spelling mistakes (including possible no mistake at all). The encoding allows the encoded words to be treated as images (there is a limit of 128 characters).  The word images are fed into a convolutional neural network that does categorization of the words.  This differs from the way CNN's commonly deal with categorization in that, instead of a one to one categorization, we treat the categorization as a way to form a tree.  This is inspired loosely by how reinforcement learning algorithms use a neural network to select action on a tree.  The categorized words that are close together cluster and form part of a tree. Thus after training a directory full of saved neural networks is created.  We also save the dictionary that forms the mapping of tree node to word. The library Jellyfish is used to form the initial separation into categories based on a subset of the words on any given node of the list of available words (passed down the tree).  This approach could be used to search for phrases also.   At present the code has been trained on a small subset of words and results are good.  Some words like big and bag can't be easily separated by the network so after the traversal is done we use Damerau Levenshtein distance.   

At spelling time the tree is traversed from root down to leaf with and the mapping dictionary is used as a mapping of leaf to word(s).   If there there is more then one word at the leaf then Dameurau Levenshtein distance is used to determine the match.  

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the code you will need jellyfish and PyTorch.

```bash
pip install jellyfish

```

See PyTorch install at [PyTorch](https://pytorch.org/get-started/locally/)

## Usage

```python

sentence_to_test = "This is a sent1nc to tset.
output = ""
for wrd in sentence_to_test.split(" "):
    # write code to spell check..

print(output)


```

## Contributing
Pull requests are welcome.   I'm not sure if I'm going to make this sample code into a framework or let it sit as is.

## Initial Development

May 2020, Christopher Schrumm



## License
[Unlicensed](http://unlicense.org/)
