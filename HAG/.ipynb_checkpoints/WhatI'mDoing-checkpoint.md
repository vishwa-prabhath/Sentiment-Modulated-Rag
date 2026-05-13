# 19/06 - Anton

1. a function to evaluate the user's prompt. (purpose: NLP based function which will tokenize the input then check against an existing DB? )

2. a function which will set up the hyper-parameters specifically temp, top_k, top_p and repetition_penalty dynamically based on a memory ( memory can be created by having an array to save the previous reply's hyper parameters.) . (purpose: I wnat this to make sure the hyper-parameters have some sort of a dynamic connection to the previous reply. by having this we have to eliminate the fact that we are allocating a new hyper parameter setting to each and every conversation point. I want this function to start with a base value and then improve on a gradient. )

# 20/06 - Vishwa

1 . a new chromadb_logger file to record the prompts and answers. And calling the function in the main ipynb file.