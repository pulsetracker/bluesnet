"""
# W266Final.py
# by James King
#
# Implements the Hitchiker's Blues Final project for W266: Natural Language Processing
#
#
#
"""
from __future__ import unicode_literals
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import unicodedata
import pronouncing

def convert_pdf_to_txt(path):
	print '\tScraping PDF'
	rsrcmgr = PDFResourceManager()
	retstr = StringIO()
	#codec = 'utf-8'
	codec = 'ascii'
	laparams = LAParams()
	device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
	fp = file(path, 'rb')
	interpreter = PDFPageInterpreter(rsrcmgr, device)
	password = ""
	maxpages = 0
	caching = True
	pagenos=set()

	for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
		interpreter.process_page(page)

	text = retstr.getvalue()

	fp.close()
	device.close()
	retstr.close()

	return text

#####################################################################################################################

def purifyText(text, separatePunctuation=True, removePunctuation=False, removeWhitespace=False):
	import re
	import string

	print '\tPurifying text'
	# Remove ligatures and similar nonsense
	t = unicodedata.normalize("NFKD",unicode(text)).encode('ascii',"ignore")

	# Remove page numbers and chapter identifiers
	pgnums = r"\n\n\d+\n\n\x0c"
	chptrs = r"Chapter \d+"
	t = text
	t = re.sub(pgnums, '', t)
	t = re.sub(chptrs, '', t)

	if removePunctuation:
		# Remove punctuation
		puncts = set(string.punctuation)
		t = ''.join(ch for ch in t if ch not in puncts)

	if separatePunctuation:
		'''Separate punctuation marks from their adjacent words so they are treated as separate tokens.'''
		puncts = set(string.punctuation)
		#textList = t.split()
		#t = ' '.join(t)
		for pMark in puncts:
			spacePMark = ' '+pMark+' '
			t.replace(pMark, spacePMark)
		

	# Remove whitespace
	if removeWhitespace:
		t = ''.join(t.split())

	return t.lower()

#####################################################################################################################

def buildDataset(filenames=['../data/guide1.pdf','../data/guide2.pdf','../data/guide3.pdf','../data/guide4.pdf','../data/guide5.pdf']):

	trilogyText = ''
	for ff in filenames:
		print 'Adding file ', ff
		try:
			thisText = convert_pdf_to_txt(ff)
			prettyText = purifyText(thisText)
			trilogyText += prettyText
		except Exception as E:
			print 'Error processing file ', ff, '\t\t', E

	outFile = open('../data/dataset.txt','w')

	print 'Writing to output file ', outFile
	outFile.write(trilogyText)
	outFile.close()

#####################################################################################################################

def pullEqSubset(startChar=0, endChar = None , separation=100, fn = '../data/dataset.txt'):

	import numpy as np

	d = open(fn).read()

	if not endChar:
		endChar=len(d)
	

	indices = np.arange(startChar,endChar,separation)

	print 'Idx: ' ,indices
	subset = [d[idx] for idx in indices]

	return subset


#####################################################################################################################

def parsewords(inputString, wordlist = '../code/wordlist.txt'):
	from math import log
	# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
	words = open(wordlist).read().split()
	wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
	maxword = max(len(x) for x in words)

	def infer_spaces(s):
		"""Uses dynamic programming to infer the location of spaces in a string
	without spaces."""

		# Find the best match for the i first characters, assuming cost has
		# been built for the i-1 first characters.
		# Returns a pair (match_cost, match_length).
		def best_match(i):
			candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
			return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

		# Build the cost array.
		cost = [0]
		for i in range(1,len(s)+1):
			c,k = best_match(i)
			cost.append(c)

		# Backtrack to recover the minimal-cost string.
		out = []
		i = len(s)
		while i>0:
			c,k = best_match(i)
			assert c == cost[i]
			out.append(s[i-k:i])
			i -= k

		return " ".join(reversed(out))

	
	return infer_spaces(inputString)

#####################################################################################################################

def checkForEnglish(wordlist):
	import enchant
	dic = enchant.Dict("en_US")
	
	if type(wordlist) == type(''):
		wordlist = list(wordlist)


	isEnglish = [dic.check(word) for word in wordlist if len(word)>1]
	
	return isEnglish

#####################################################################################################################
def getNumSyllables(word): 
	'''Returns the number of syllables in the string word.
	Note:  word must be in the cmudict.  If it isn't, function will return 0 syllables.

	An adaptation of stackoverflow.com/questions/5087493/to-find-the-number-of-syllables-in-a-word
	to make it both readable and make it work
	'''
	import curses 
	from curses.ascii import isdigit 
	import nltk
	from nltk.corpus import cmudict 
	d = cmudict.dict() # a dictionary of pronounciation

	syllable_count = 0
	if word.lower() in d.keys():
		for sounds in d[word.lower()][0]:  #d['astronomy'] returns [u'AH0', u'S', u'T', u'R', u'AA1', u'N', u'AH0', u'M', u'IY0']
			for s in sounds:
				s_ascii = str(s)
				
				if isdigit(s_ascii[-1]):		#syllables have a digit on them
					syllable_count += 1

	#print d[word.lower()]
	return syllable_count

#####################################################################################################################

def getRhymes(word):
	
	rhymes = pronouncing.rhymes(word)

	return rhymes


#####################################################################################################################

def buildNGrams(fullText, n=3, shortSample = False, printSample = False):
	from nltk import ngrams

	if shortSample:
		fullText = " ".join(fullText.split()[0:10000])

	nGrams = ngrams(fullText.split(), n) # Returns a list of tuples
	
	if printSample:
		for g in nGrams[0:10]:
		  print g


	return nGrams
#####################################################################################################################
import pickle
def getNGramAdams(buildNewDataSet=False):
	import os
	
	if buildNewDataSet or ('dataset.txt' not in os.listdir('../data')):
		print 'dataset.txt not found.  Rebuilding'
		buildDataset()

	text = open('../data/dataset.txt').read()

	ngGen = buildNGrams(text, shortSample=True, printSample=False)

	print 'Writing n-grams to pickle'
	ng = [n for n in ngGen]

	pickle.dump(ng, open( '../data/ngrams.txt','w'))
	
#####################################################################################################################

def readTheBlues():
	pass

	# open the file
	# make a list with a song as each element
	# for each song, count verses & attempt to find song structure
	# for each verse, load lines
	# for each line, build an ngram.

class bluesTune():

	""" Example song:

	  Aker   1       Akers, Garfield
	\C Text transcribed from discography listed in and edited for publication in
	\C Michael Taft, \iBlues Lyric Poetry: An Anthology\r. New York: Garland
	\C Publishing, Inc., 1983. See also, Michael Taft, \iBlues Lyric Poetry: A
	\C Concordance\r. New York: Garland Publishing, Inc., 1984.
	\C    title: Cottonfield Blues-+-Part 1
	\C    place and date: Memphis, c. 23 Sept. 1929
	\C    record numbers: (M-201- ) Vo-1442 OJL-2

	I said look a-here mama :
	 what in the world are you trying to do
	You want to make me love you : 
		you going to break my heart in two

	I said you don't want me : 
		what made you want to lie
	Now the day you quit me fair brown :
		 baby that's the day you die

	I'd rather see you dead :
		 buried in some cypress grove
	Than to hear some gossip mama :
		 that she had done you so

	It was early one morning :
		 just about the break of day
	And along brownskin coming :
		 man and drove me away

	Lord my baby quit me :
		 she done set my trunk outdoors
	That put the poor boy wandering :
		 Lord along the road

	I said trouble here mama :
		 and trouble everywhere you go
	And it's trouble here mama :
		 baby good gal I don't know

		"""

	def __init__(self, rawText=None):

		self.rawText = "\L"+rawText.strip()
		self.artist = ''
		self.title = ''
		self.time = ''
		self.place = ''
		self.record_numbers = ''
		self.fullHeaderText = ''
		self.fullLyricText = ''
		self.shorthand_code = ''

		self.format = ''  # Something like AABA
		self.stanzas = []  # one element in the array for each two lines (1 stanza) of the song

		if self.rawText:
			self.parseSong()

	def __str__(self):
		outText = '\n'
		for stz in self.stanzas:
			outText+=stz.replace('::','\n').replace(':','\n\t')+'\n\n'
		#return '\n\n'+self.fullLyricText.replace('::','\n2').replace(':','\n\t')
		return outText

	def sents(self):
		sent = []
		for stz in self.stanzas:
			for word in stz.split():
				sent.append(word)
			sent.append(':::')
		#return [stz.split() for stz in self.stanzas]
		return sent


	def parseSong(self):
		if self.rawText: 

			splitRaw = self.rawText.split('\\')

			self.shorthand_code = " ".join(splitRaw[0].strip().split()[0:2])
			self.artist = " ".join(splitRaw[0].strip().split()[2:])

			for ss in self.rawText.split('\n\n'):
				if ss.startswith('\\'):
					self.fullHeaderText+= ss+"\n"
				else:
					self.fullLyricText += ss+"\n"

	
			# join each pair of lines into a stanza with lines separated by ::
			'''	I said trouble here mama : and trouble everywhere you go
				And it's trouble here mama : baby good gal I don't know

					becomes

				I said trouble here mama : and trouble everywhere you go :: And it's trouble here mama : baby good gal I don't know



				'''

			stack = []
			intText = self.fullLyricText.split('\n')

			for ll in list(enumerate(intText)):
				print 'll: ', ll
				if ll[0]%2==0:
					stack.append(ll[1])
				else:
					self.stanzas.append(stack.pop()+' :: '+ll[1])


def loadBluesSongs(process=True):
	"""
	process:  Whethter to process out caps and weird characters
	"""

	files = ['../data/bluesLyrics/1409/blues1.1409', 
				'../data/bluesLyrics/1409/blues2.1409']

	songs = []

	for ff in files:
		try:
			thisFile = open(ff).readlines()
			thisBlock = "\n".join(thisFile)

			
			thisSongs = thisBlock.split('\L')

			for song in thisSongs:
				if process:
				 	song = song.replace('*','')
				 	song = song.lower()
	
				songs.append(bluesTune(rawText = song))

		except Exception as e:
			print 'Error importing file ', ff, ' || ', e



	return songs

def buildEmbeddings():
	from gensim.models import Word2Vec
	#from nltk.corpus import brown, movie_reviews, treebank
	#>>> b = Word2Vec(brown.sents())
	#>>> mr = Word2Vec(movie_reviews.sents())
	#>>> t = Word2Vec(treebank.sents())
 
	#>>> b.most_similar('money', topn=5)
	#[('pay', 0.6832243204116821), ('ready', 0.6152011156082153), ('try', 0.5845392942428589), ('care', 0.5826011896133423), ('move', 0.5752171277999878)]

	songs = loadBluesSongs()

	allSents = [ss.sents() for ss in songs]
	allText = "\n".join([ss.fullLyricText for ss in songs])
	em = Word2Vec(allSents)

	
def genLineBigram(in_songs = None, startWord=None, num=15):
	import random
	import nltk

	if not in_songs:
		songs = loadBluesSongs()
	else:
		songs = in_songs

	songWords = " ".join([ss.fullLyricText for ss in songs]).split()
	bigrams = nltk.bigrams(songWords)
	cfdist = nltk.ConditionalFreqDist(bigrams)
	cpdist = nltk.ConditionalProbDist(cfdist, nltk.ELEProbDist)

	if startWord == None:
		word = random.choice(cpdist.keys())
	else:
		word = startWord

	outLine = ''
	for i in range(num):
		outLine += word + ' '
		word = cpdist[word].generate()

	#print outLine

	return outLine,cpdist

#####################################################################################################################


def genLine(in_songs = None, startContext=None,n=2, num=15):
	import random
	import nltk

	if not in_songs:
		songs = loadBluesSongs()
	else:
		songs = in_songs

	songWords = " ".join([ss.fullLyricText for ss in songs]).split()
	# bigrams = nltk.bigrams(songWords)
	ngrams = nltk.ngrams(songWords, n)
	#condition_pairs = (((w0, w1), w2) for w0, w1, w2 in ngrams)
	condition_pairs = [((w[0:-1]),w[-1]) for w in ngrams]
	cfdist = nltk.ConditionalFreqDist(condition_pairs)
	cpdist = nltk.ConditionalProbDist(cfdist, nltk.ELEProbDist)

	if startContext == None:
		context = tuple([random.choice(cpdist.keys()) for nn in range(n-1)])
	else:
		context = startContext

	# All the start words should be a tuple, but users are unlikely to enter it that way
	if type(context)==type(''):
		context = tuple(context.split())


	outLineList = []
	for i in range(num):
		print i, num, outLineList #context
		outLineList += list(context)

		try:
			newWord = cpdist[context].generate()
		except:
			newWord = random.choice(cpdist.keys())

		newContext = (outLineList[-(n-1)],newWord)
		context=newContext

		#print outLineList
	#return " ".join(outLineList)
	return outLineList

#####################################################################################################################
if __name__ == "__main__":
	main()