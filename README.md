


# An Investigation of the Bible: Generating Multi-faceted Character Profiles



## (1) General Submission Guidelines

* Working title: An Investigation of the Bible: Generating Multi-faceted Character Profiles

* Team members: Felix Munzlinger, Johannes Klüh, Aileen Reichelt & Sven Leschber

* Mail addresses: felix.munzlinger@gmail.com, joe.klueh@gmx.de, reichelt@cl.uni-heidelberg.de, rz227@uni-heidelberg.de

* Existing Code Fragments:
	* YAKE! (https://www.sciencedirect.com/science/article/pii/S0020025519308588) - to do the keyword extraction
	* Austin Taylor (http://www.austintaylor.io/d3/python/pandas/2016/02/01/create-d3-chart-python-force-directed/) - to create a force directed network diagram to outline a relation graph
	* NeuralCoref (https://github.com/huggingface/neuralcoref) - to get co-reference resolution

* Utilized libraries: please find the [requirements.txt](https://github.com/SvenElyes/Textanalytics/blob/main/requirements.txt) file in our repository

* Contributions: In addition to our commit history, you may also refer to our GitHub issues. All members contribute to planning of the project as well as the write-up of the reports. Also, please note that Aileen commited changes as both "aileen-reichelt" and "Aileen Reichelt".

## (2) Project State

Planning State: 

Our project consists of a variation of tasks which need to be completed in order to create elaborate character profiles. As we have shown in our project proposal, there are three major topics:

	(i) Keyword extraction
	
	(ii) Character relations
	
	(iii) Character traits
	
As discussed with our coordinator John Ziegler, we will first focus on the character relations and keywords to get to know the dataset in more detail. Also, this could be evaluated by using existing data concerning the bible. Therefore, we may be able to evaluate how good our results are just by comparing our findings on existing literature, e.g. from the field of theology.

As for the planning of our next steps and goals, we defined issues to be further pursued. Those issues have been added to the GitHub Issue board to keep everything in one platform. The naming convention over there is: `topic - pipeline - issue`.

After feedback from our coordinator, we also discussed that it would be great to find character relations through multiple approaches. First, we want to evaluate which characters are mentioned in the same context. As the bible is split into books, chapters and verses we can build a three-stage pipeline. Characters which are mentioned in the same verse persumably have a close relation, characters which are mentioned in the same chapter persumably have a medium close relation and characters which are mentioned in the same book persumably have a low relation to each other. 

However, this does not give any indication on how friendly the relation between two characters is. As John indicated, we could use a list of words which are marked as "positive" and "negative" as a form of basic sentiment analysis to evaluate how their relation is displayed. Felix and Sven will cover this topic. 

But there many more more ways to determine the relation of two characters. Assuming that we have extracted keywords to any verse, we could assign those keywords to any character mentioned in this verse. Then, we could do a clustering of those characters to evaluate which are closest to each other in regards of similar keywords. The clustering may be done using the Faiss implementation by Facebook. By doing so, we could determine which characters are mentioned in the same context content-wise, rather than only investigating which occur in the same text. The keyword extraction is done by Aileen. The clustering is a subsequent task, which will be assigned afterwards, depending on the workload currently at hand.

Furthermore, there are sentences like "Jesus broke the bread. He shared it among his 12 apostles." He is clearly referencing Jesus, however, it only is an implicit character mention which would not be recognized if we where only looking for names. Therefore, we need to work with co-references, to extract those references. This topic will be convered by Johannes. 

As of now, we will put the task of the character traits at the end of our current development pipeline. They present a whole now topic and may be a subsequent task we will approach once we have determined the relations and keywords. 

## (3) Data Analysis

The bible is well documented. So well, there are several GitHub Repositories which have formatted its content to an XML/JSON/SQL state. In [this](https://github.com/bibleapi/bibleapi-bibles-json) repository, the bible is available in multiple languages, from which we will use the English version. The document structure of the json file follows an easy structure which will make it accessible for use. Example: 

```shell
{"chapter":1,"verse":1,"text":"In the beginning God created the heavens and the earth.","translation_id":"ASV","book_id":"Gen","book_name":"Genesis"}
{"chapter":1,"verse":2,"text":"And the earth was waste and void; and darkness was upon the face of the deep: and the Spirit of God moved upon the face of the waters","translation_id":"ASV","book_id":"Gen","book_name":"Genesis"}
{"chapter":1,"verse":3,"text":"And God said, Let there be light: and there was light.","translation_id":"ASV","book_id":"Gen","book_name":"Genesis"}
...
```				
For this reason, we don't really have to do much data pre-processing, as it is already in a quite confortable state and tasks such as tokenization are mostly included in the libraries we use, e. g. YAKE. The only thing we do in terms of processing
is creating a pandas `DataFrame`, which has the same structure as our json objects in order to make use of pandas' functions.

Our `DataFrame` has 31,102 rows, each containing a verse of the bible. There are 66 books in the bible. In general, a lot of the information about the data set is equivalent to information regarding the bible in general. For example: There are  1,189 chapters in the Bible (on average, 18 per book).

Further information: https://en.wikipedia.org/wiki/Chapters_and_verses_of_the_Bible
 
Example:
This is the .head() output of the `DataFrame` containing the bible:
```shell
   chapter  verse                                               text translation_id book_id book_name
0        1      1  In the beginning God created the heavens and t...            ASV     Gen   Genesis
1        1      2  And the earth was waste and void; and darkness...            ASV     Gen   Genesis
2        1      3  And God said, Let there be light: and there wa...            ASV     Gen   Genesis
```
## (4) Current Code State

Progress:  
The written code progress is in an early state. Since the team members will be able to dedicate more time to the project the following weeks, more progress can be expected soon.  

Code structure:  
All the information which we will aquire in the process will be gathered in a Python class for character relations created by us. Consequently, we also introduce a character class, which will later be used to store a character's traits. After extracting all the information needed and storing it in instances of these classes, we will need a function that returns an instance of the relations class for two given instances of the character class. This information, together with the two characters in reference, is stored in an array. After determining this array, we will be able to apply it to a force directed network diagram. The mentioned application for this works with pandas, hence our use of this data structure.

There already exists a [class](https://github.com/SvenElyes/Textanalytics/blob/main/src/dataloader.py) for loading the bible data into a pandas `DataFrame` and first versions of the [character](https://github.com/SvenElyes/Textanalytics/blob/main/src/data/character.py) and [relation](https://github.com/SvenElyes/Textanalytics/blob/main/src/data/relation.pyy) classes. Note that a relation may not be equally strong or positive for both characters, which makes it necessary to create two relations for two characters: `Char1` &rarr; `Char2` and `Char2` &rarr; `Char1`.

Both classes have to be extended by further methods to process the information stored in their attributes.

So that we are able to save our progress in creating characters and relations we use the [pickle](https://docs.python.org/3/library/pickle.htm) module, which is provided by the standard python library.

The handling of the .pkl files and the character and relation objects is done by the [Pickle Handler](https://github.com/SvenElyes/Textanalytics/blob/main/src/pickle_handler.py) . It will be able to save and load characters from those files.

On top of this, the [Relation Creator](https://github.com/SvenElyes/Textanalytics/blob/main/src/relation_creator.py) will use the picklehandler and [Keyword Extractor](https://github.com/SvenElyes/Textanalytics/blob/main/src/keyword_extractor.py ) to create all the characters and their attributes from the distilled csv and save it to .pkl files.

One of the main challenges is recognizing characters in the bible and to save all characters for each verse. This is done in the [Named Entities](https://github.com/SvenElyes/Textanalytics/blob/main/src/named_entities.py)
file, in which our bible csv is extended by a collumn containing the characters which appear in each row.

To assess our relations we use the [preprocess emotion](https://github.com/SvenElyes/Textanalytics/blob/main/src/preprocess_emotion.py) file, which looks at  words which often appear in the same space as two characters and tries to assign a positive or negative connotaiton towards those words . This is done via the the two .txt files with one, containing a list of [negative words](https://github.com/SvenElyes/Textanalytics/blob/main/src/neg_bag_of_word.txt) and the other one containing a list of  [positive words](https://github.com/SvenElyes/Textanalytics/blob/main/src/pos_bag_of_word.txt).

The  [eval_graph](https://github.com/SvenElyes/Textanalytics/blob/main/src/eval_graph.py) file will handle the main part of the workflow by  clustering our results and creating the graph. It uses the Keyword Extractor indirectly via the PickleHandler and creates the distilled csv (containing a CSV with distinct characters and the relationships they have with each other)  from which we will create the character objects.


Code quality:  
In order to ensure coherent code between all team members, we have agreed to follow the PEP-8 style guidelines, most importantly in regards to documentation. Since all aspects from naming conventions to spacing and docstrings are covered by this, we have not set any additional rules.
