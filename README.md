# comp441projectMemNN
implementation of MemNN(http://arxiv.org/pdf/1410.3916.pdf)

Results and explanation of the mechanism:https://drive.google.com/file/d/0B_atXUU8Ks24SmxHRE5iR0g1Ync/view?usp=sharing


I have started with baselines and the simplest form of my project was 1-word answer creating according to 15 sentence story and questions. The question is directly about the first related memory.
The MemNN consist of 4 Modules(I:Input,G:Generalisation,O:Output,R:Response). I Module converts a single sentence to BoW representation(vector) as described by paper. R Module is where the memories(BoW representations of each sentence) are stored. The R Module is a Matrix of 19x14 where 19 is number of words in dictionary and 14 is memory capacity. O and R modules are responsible for generating answers to questions, they only activate when there is a question. The O module takes the BoW representation of question(Bow is created by I module) and it compares it to each of the memories(which are also stored as BoW by Memory Matrix), this comparison is done with a specific cost function (X.U’.U.Y) by which X is question and Y is each of the memories. U is n x D matrix, where n indicates dimension of hidden units and D is is number of features(|W| by O1 and|3W| by R)The paper presented argmax and margin loss for selection and training of best matching memory. But I have used softmax and softloss and trained to get probabilities. Instead of using a single memory entry vector for Y, I have used the entire memory matrix. So my cost function outputs an array of values, which correspond to probabilities of each memory location as a match. The R module is also similar, it uses the same cost function but it is trained seperately. The X is the question appended with best matching memory. Y is BoW representation of each word. The places for words of question, matching memory and output are different by BoW vector, which means it is 3W length vector(W is number of words). It corresponds to adding each word with labels such as Q:Where, Q:is, Q:Sandra, M:Sandra, M:went, M:to, M:kitchen, ANS:kitchen.
	First the data of training text and test text are processed and added to a matrix of Strings. Then the dictionary is initialised with String -> Int:(0 initially)  to posses all of the words in the data but without concatenated punctuation(such as dots,question marks, or \t characters). Then the Memory Matrix(19x14) is initialised.
	The modules are implemented as Julia functions and the cost functions are implemented as Knet models. 
The I module takes a String, removes undesired characters(such as comma, dot) and creates a vector of BoW representation, by creating a copy of dictionary and incrementing the values of each word by the input String.
The G module simply saves the new sentence vector to its appropriate place in Memory Matrix according to the memory place counter.
The O module takes Memory Matrix, , question sentence as BoW ,the location of best matching memory as one hot vector , a boolean for indicating whether is it trainmod, a Knet model as o1_costmodel.
The R module  takes the most related memory(BoW) , question(BoW), the answer word(BoW), dictionary, trainmod(boolean), ,r_costmodel(Knet model). The R module appends the question vector first with new memory, than with a all 0-valued dictionary BoW vector, so it becomes |D| = 3 |W|. It also creates BoW representation of each word in dictionary and forms a matrix from them, the cost function ranks the matches of (x, memory1) with (w) for w is an element of W(dictionary) for all W. Feeding cost model with a matrix instead of separate vectors, provides flexibility to use probabilities with softmax. 
The O module and R module don’t access to cost models directly. They run train_o1 and train_r functions to forward values, backpropagate and to update according to the specified loss function and expected output.
o1_cost is a Knet Model/function, which takes  new input(question,X) and Memory Matrix(Y) as BoW representations. And trained  (X.U’.U.Y) with softmax classifier and softloss.
r_cost is a Knet Model/function, which takes the question appended with memory1 (BoW) and dictMatrix(every word has a onehot representation). And trained  (X.U’.U.Y) with softmax classifier and softloss.

The first for loop by mainflow part is for epochs and the first inner for loop is for traversing on  training data and training the models. Second inner for loop is for testing the model by test data. The statistics are printed for each epoch.

With additional commits, MemNN is modified such that the embedded word representations are large enough to posses slots for the 2.supporting fact and the absolute time vector.

The 2supfactMemNN is for question answer pairs with two supporting facts. The O module finds most related first supporting fact according the question, than it finds the most related second supporting facts according to the question and first supporting fact. The cost functions for the first and second supporting facts are different. R module generates predicted answer according to question and 2 supporting facts.
