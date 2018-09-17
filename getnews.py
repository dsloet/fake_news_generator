# -*- coding: utf-8 -*-


import newspaper
import os


class Getnews:
    def __init__(self, source, output):
        """Getnews gets news from a given newssite and stores it in a text file.
        Args:
            source (string): "The link to the newssite. Format: 'http://cnn.com'"
            output (string): "The file where titles need to be stored. Format: 'output.txt'"

        Examples:
            >>> news = Getnews('http://cnn.com', 'output.txt')
            >>> news.getnews()
        
        
        """
        path = os.getcwd()
        
        self.source = source
        self.output = os.path.join(path, output)

    def getnews(self, memoize=False):
        """
        The getnews() instance accepts a boolean for memoize.

        Args:
            memoize=True: Memorizes the already seen articles
            memoize=False: Extracts all articles even if it has seen them before
        
        Examples:
            >>> news = Getnews('http://cnn.com', 'output.txt')
            >>> news.getnews(memoize=True)

        """

        self.titles = []
        print("input newssite = " + self.source)
        print("output file = " + self.output)
        paper = newspaper.build(self.source, memoize_articles=memoize)
        for articles in paper.articles:
            
            try:
                articles.download()
                articles.parse()
                print(articles.title)
                self.titles.append(articles.title)
            except:
                pass
        self.storenews()

    def storenews(self):
        for title in self.titles:
            with open(self.output, 'a+') as f:
                f.writelines(title + '\n')

