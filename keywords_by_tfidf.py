import os
import numpy as np
import sklearn.feature_extraction.text as sktext

##чтобы программа работала, необходимо установить
##модули numpy, scipy, pandas(возможно ?) и scikit-learn

class mean_tfidf_by_cat():
    '''аргумент на вход - массив из путей к папкам с категориями
(в формате Unix, т.е. ./example/example/ Питону 3.6 точно пофиг
на операционную систему)'''
    def __init__(self, cat_paths):
        ##в TfidfVectorizer нужно добавить аргументом массив со
        ##stopwords, чтобы лучше работало
        self.vectorizer = sktext.TfidfVectorizer(input = 'filename')
        self.alltexts_dict = dict()
        ##складываем имена файлов в словарь по категориям:
        for i in cat_paths:
            self.alltexts_dict[i.split('/')[-2]] = [i + j for j in os.listdir(i)]
        alltexts_list = []
        for k in self.alltexts_dict:
            alltexts_list += self.alltexts_dict[k]
        self.vectorizer.fit(alltexts_list)
        self.features = self.vectorizer.get_feature_names()

    def get_keywords(self, category, n_keywords):
        '''аргументы на вход: имя категории (название папки с категорией,
не путь к ней; количество ключевых слов, которые мы ходим выделить'''
        category_tfidf = self.vectorizer.transform(self.alltexts_dict[category])
        category_tfidf = np.array(category_tfidf.mean(axis=0))[0]
        top_n = get_top_n(category_tfidf, n = n_keywords)
        keywords = {self.features[k]:v for k,v in top_n.items()}
        return keywords

def get_top_n(lst, n):
    indices = [i for i in range(len(lst))]
    indices = sorted(indices, key = lambda x: lst[x], reverse = True)
    topn = {i:lst[i] for i in indices[:n]}
    return topn

##пример использования - можно закомментить
if __name__ == '__main__':
    a = mean_tfidf_by_cat(('./training/detective/','./training/horror/','./training/spy/'))
    print(a.get_keywords('detective',100))
    print(a.get_keywords('horror',100))
    print(a.get_keywords('spy',100))
