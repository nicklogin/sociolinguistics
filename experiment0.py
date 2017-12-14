import os
import numpy as np
import sklearn.feature_extraction.text as sktext
import statistics

##чтобы программа работала, необходимо установить
##модули numpy, scipy, pandas(возможно ?) и scikit-learn

class mean_tfidf_by_cat():
    '''аргумент на вход - массив из путей к папкам с категориями
(в формате Unix, т.е. ./example/example/ Питону 3.6 точно пофиг
на операционную систему)'''
    def __init__(self, cat_paths):
        ##в TfidfVectorizer нужно добавить аргументом массив со
        ##stopwords, чтобы лучше работало
        stopwords = ['я','ты','он','она','оно','они','не','в','п','над','под','от','к',
                 'да','нет','перед','через','после','спереди','сзади','из-за','и','на',
                 'а','но','для','до','как','какой','мочь','можно','или','если','это',
                     'так','что','вы','все','эта','эти','быть', 'кто', 'то', 'по', 'за', 'ли',
                     'com', 'https','бы', 'еще', 'там','же','нибудь','тоже', 'много', 'идти']
        self.vectorizer = sktext.TfidfVectorizer(input = 'filename',stop_words = stopwords)
        self.alltexts_dict = dict()
        ##складываем имена файлов в словарь по категориям:
        for i in cat_paths:
            self.alltexts_dict[i.split('/')[-2]] = [i + j for j in os.listdir(i)]
        self.alltexts_list = []
        for k in self.alltexts_dict:
            self.alltexts_list += self.alltexts_dict[k]
        self.vectorizer.fit(self.alltexts_list)
        self.features = self.vectorizer.get_feature_names()

    def get_keywords(self, category, n_keywords):
        '''аргументы на вход: имя категории (название папки с категорией,
не путь к ней; количество ключевых слов, которые мы ходим выделить;
если имя категории - *, то находим самые частотные слова дял общего
корпуса'''
        if category == '*':
            category_tfidf = self.get_average_scores(self.alltexts_list)
        else:
            category_tfidf = self.get_average_scores(self.alltexts_dict[category])
        top_n = get_top_n(category_tfidf, n = n_keywords)
        keywords = {self.features[k]:v for k,v in top_n.items()}
        return keywords

    def get_average_scores(self, textlist):
        average_tfidf = self.vectorizer.transform(textlist)
        average_tfidf = np.array(average_tfidf.mean(axis=0))[0]
        return average_tfidf

    def get_corpora_average(self, keywords = []):
        average_matrix = self.get_average_scores(self.alltexts_list)
        if keywords:
            if isinstance(keywords,dict):
                keyword_indices = get_indices([i for i in keywords.keys()],self.features)
            elif isinstance(keywords,list) or isinstance(keywords,tuple) or isinstance(keywords,set):
                keyword_indices = get_indices(keywords,self.features)
            else:
                print('keywords argument must be either dict(), list(), tuple() or set() object!')
                raise Error
            average_scores = {v:average_matrix[k] for k,v in keyword_indices.items()}
            return average_scores
        else:
            return average_matrix

    def keyword_chi_square(self, d1, d2):
        '''на вход поступают уже готовые словари ключевых слов для
подкорпуса и всего корпуса'''
        stat = text_len_stat(self.vectorizer.build_tokenizer())
        average_text_length = stat.get_average_len(self.alltexts_list)
        idfs = tuple(self.vectorizer.idf_)
        indices = get_indices(d1, self.features, reverse = True)
        d1 = {k:v*average_text_length/idfs[indices[k]] for k,v in d1.items()}
        d2 = {k:v*average_text_length/idfs[indices[k]] for k,v in d2.items()}
        return sum([((d1[k]-d2[k])**2)/d1[k] for k in d1])

class text_len_stat():
    def __init__(self, tokenizer):
        self.tokenize = tokenizer
        pass

    def get_average_len(self, texts):
        average_length = 0
        for t in texts:
            f = open(t,'r',encoding='utf-8').read()
            average_length += len(self.tokenize(f))
        f = ''
        average_length /= len(texts)
        return average_length

    def get_stats(self, texts):
        lens = []
        for t in texts:
            f = open(t,'r',encoding='utf-8').read()
            lens.append(len(self.tokenize(f)))
        return {'var':statistics.pvariance(lens),'stdev':statistics.pstdev(lens),'max':max(lens),'min':min(lens)}
        
     
def get_top_n(lst, n):
    indices = [i for i in range(len(lst))]
    indices = sorted(indices, key = lambda x: lst[x], reverse = True)
    topn = {i:lst[i] for i in indices[:n]}
    return topn

def get_indices(values, ar, reverse=False):
    indices = dict()
    if reverse:
        for v in values:
            indices[v] = ar.index(v)
    else:
        for v in values:
            indices[ar.index(v)] = v
    return indices

##пример использования - можно закомментить
if __name__ == '__main__':
    a = mean_tfidf_by_cat(('./MPGU/','./training/horror/','./MGTU/'))
    most_frequent = a.get_keywords('*',100)
    print('самые частотные:')
    print([i for i in most_frequent])
    MPGU = a.get_keywords('MPGU',200)
    MPGU = {i:MPGU[i] for i in MPGU if i not in most_frequent}
    print('МПГУ')
    print(MPGU)
    average_1 = a.get_corpora_average(MPGU)
    MGTU = a.get_keywords('MGTU',200)
    MGTU = {i:MGTU[i] for i in MGTU if i not in most_frequent}
    print('МГТУ')
    print(MGTU)
    average_2 = a.get_corpora_average(MGTU)
    comparision = [(MPGU, average_1), (MGTU, average_2)]
    for i in comparision:
        print(a.keyword_chi_square(i[0],i[1]), len(i[0]))
    stat = text_len_stat(a.vectorizer.build_tokenizer())
    print(stat.get_average_len(a.alltexts_list))
    print(stat.get_stats(a.alltexts_list))
