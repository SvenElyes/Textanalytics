# Textanalytics

https://github.com/jomazi/Python-Default

(1) General Submission Guidelines

• Title: An Investigation of the Bible:Generating Multi-faceted CharacterProfiles

• Team Members: Felix Munzlinger, Johannes Klüh, Aileen Reichelt & Sven Leschber

• Mail Addresses: felix.munzlinger@gmail.com, joe.klueh@gmx.de, reichelt@cl.uni-heidelberg.de, rz227@uni-heidelberg.de

• Existing Code Fragments: Jake! (https://www.sciencedirect.com/science/article/pii/S0020025519308588) - to do the keyword extraction, Austin Taylor (http://www.austintaylor.io/d3/python/pandas/2016/02/01/create-d3-chart-python-force-directed/) - to create a force directed network diagram to outline a relation graph,

• Utilized libraries: Please find the requirements.txt file attached to our repository

(2) Project State

Planning State: 
Our projects presents a variation of tasks which need to be completed in order of determining a well elaborated character profile. In fact, as we have shown in our project proposal there are several major topics such as:

	(i) Keyword extraction
	
	(ii) Character relations
	
	(iii) Character traits
	
As discussed with our coordinator John Ziegler, we will first focus on the character relations to get to know the dataset in more detail. Also, this could be evaluated be using existing data concerning the bible. The bible is the most printed book in the world and there is even a subject to study if someone is interested in christian religion - theology. Therefore, we may be able to evaluate how good our result are just by probing our findings on existing literature. We already defined issues for this topic, to be further pursued. Those issues have been added to the GitHub Issue board in order of keeping it at one platform. The naming convention over there is: topic - pipeline - issue

As we have also discussed it would be great to find character relations through multiple approaches. First, we want to evaluate which characters are mentioned in the same context. As the bible is split into books, chapters and verses we can make build an 3 stage evaluation. Characters which are mentioned in the same verse persumably have a high relation, characters which are mentioned in the same chapter persumably have a medium relation and characters which are mentioned in the same book persumably have a low relation. However, this does not give any indication on how friendly the relation between two characters is. As John indicated, we could use a list of words which are marked as "possitive" and "negative" to evaluate how their relation is displayed. Felix and Sven will cover this topic. But there may more more ways to determine the relation of two characters. Assuming that we have extracted keywords to any verse, we could assign those keywords to any character mentioned in this verse. Then, we could do a clustering of those characters to evaluate which are closest to each other. The clustering may be done using the Faiss implementation by Facebook. By doing so, we could determine which characters are mentioned in the same context. Rather than only investigating which occur in the same text. The keyword extraction is done by Aileen. The clustering is a subsequent task, which will be assigned afterwards, depending on the workload currently at hand. Nevertheless, there are sentences like "Jesus broke the bread. He splitted it throughout his 12 apostles." He is clearly referencing Jesus, however, it only is an implicit character mention which would not be recognized if we where only looking for hard names. Therefore, we need to work with co-relations, to extract those references.(which may in practice extend over a single verse!) This topic will be convered by Johannes. 
All the information, which we will aquire in the process will be put into a relation specific class. This class formulates the relation between 2 characters. Consequently we also introduced a character class, which will later also be used to store its character traits. After extracting all the information needed we will have a method that formulates the state of the relation between two characters. This information is stored in an array in conjunction with the two characters in reference. After determining this array we will be able to apply it to a force directed network diagram. The mentioned application works with pandas and therefore be perfectly fine to use with our data structure. 

As of now, we will put the character traits at the end of our current development pipeline. This presents a hole now topic and may be a subsequent task we are approaching, once we have determined the relations and keywords. 

(3) Data Analysis

The bible is well documented. So well, there are actual GitHub Repositories, which have put its content to an XML/JSON/SQL state. A repository like this is https://github.com/bibleapi/bibleapi-bibles-json. Here, the bible is available in multiple languages, from which we will use the english language. The document structure of the json file follows an easy structure which will make it well accessable to use:
{"chapter":1,"verse":1,"text":"In the beginning God created the heavens and the earth.","translation_id":"ASV","book_id":"Gen","book_name":"Genesis"}
{"chapter":1,"verse":2,"text":"And the earth was waste and void; and darkness was upon the face of the deep: and the Spirit of God moved upon the face of the waters","translation_id":"ASV","book_id":"Gen","book_name":"Genesis"}
{"chapter":1,"verse":3,"text":"And God said, Let there be light: and there was light.","translation_id":"ASV","book_id":"Gen","book_name":"Genesis"}
...
				
For this reason we don't really have to do much data pre-processing, as it is already in a quite confortable state. The only thing we do in terms of processing
is creating a Pandas DataFrame which has the same structure as our json objects. Because of this, we create a CSV file from which we will perform all further actions and analysis.


##################
Basic Statistic fehlt hier noch
Our DataFrame has 31102 rows, each containing an individual verse of the Bible. 
There are 66 individual books of the bible.


##################

(4) Current Code State
