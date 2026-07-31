# 5th place solution

Our code is [here](https://www.kaggle.com/code/gromml/arc-prize-2024-poohai-solution).

Basically, our solution consists of 3 ideas:

1) **Ensembling different solutions.**

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2F83488802523e694d5ab4ce5984e0e0f6%2Fensembling__.png?generation=1733569928133209&alt=media" width="500">

2) **Applying different postprocessing filters.**

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2F60ed2df7d537be28c08c220165c356e4%2Fpostprocessing_.png?generation=1733569982701177&alt=media" width="500">

We noticed that an algorithm can make typical mistakes. For instance, a genetic algorithm tends to produce redundant vertical or horizontal lines.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2Fde875adfed78ba8c904cdc6301f4c0b3%2Fgen1_.png?generation=1733570064376646&alt=media" width="500">

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2F5b5c27d16797fc2dbd2ef7d164eca553%2Fgen2_.png?generation=1733570076948928&alt=media" width="500">

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2F433d8a9c106d0945b33cb5bd843dd0c8%2Fgen3_.png?generation=1733571013703337&alt=media" width="500">

Also, another typical mistake is a wrong shape.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2F5455fe567ae591c94fe1a68f9e3fe805%2Fgen4_.png?generation=1733570102883731&alt=media" width="500">

So, you can implement as many postprocessing filters as you can to cover the most common mistakes of algorithms in your ensemble.

3) **Brute-force.** Tasks can be sorted by their identifiers, and the order remains the same across different submissions.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3365741%2Ffd5064ef9910eaeeb01c31916a331524%2Forder_.png?generation=1733569963472044&alt=media" width="500">

To identify the 26 tasks solved by the [well-known 26 notebook](https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions), you need no more than 100 submissions. Once the 26 tasks are identified, you can try new algorithms one by one (checking whether they are able to solve at least one new task, that is one of the 74 remaining tasks). Once a new task is found, you can identify its ordinal number with the help of binary search. Also, you can leave all other tasks to the strongest algorithm.

**Our summary:**

* A very efficient way to improve an ensemble was to identify tasks solved by the ensemble, and then try to solve other tasks by other algorithms
* It seems that LLMs could be added to our ensemble, but fine-tuning is needed
* Selecting the right attempt (the correct answer) is difficult, while removing wrong attempts is easier
* An algorithm can make typical mistakes (like the genetic algorithm often produces redundant lines, and you can create the corresponding postprocessing filters)

Congratulations to all the winners, and thanks to the host team for organizing the competition! Looking forward to ARC Prize 2025!