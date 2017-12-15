import os
import re
import numpy as np
import sklearn.feature_extraction.text as sktext
import statistics
import csv

##чтобы программа работала, необходимо установить
##модули numpy, scipy, pandas(возможно ?) и scikit-learn

class mean_tfidf_by_cat():
    '''аргумент на вход - массив из путей к папкам с категориями
(в формате Unix, т.е. ./example/example/ Питону 3.6 точно пофиг
на операционную систему)'''
    def __init__(self, cat_paths):
        ##в TfidfVectorizer нужно добавить аргументом массив со
        ##stopwords, чтобы лучше работало
        rus_dict = csv_parse('freqrnc2011.csv')
        stop_pos = {'conj', 'anum', 'intj','advpro','spro','apro', 'pr', 'num', 'part'}
        stopwords = [i['Lemma'] for i in rus_dict if i['PoS'] in stop_pos]
        stopwords += ['vk','com','https','photo']
        self.vectorizer = sktext.TfidfVectorizer(input = 'filename',stop_words = stopwords)
        self.alltexts_dict = dict()
        ##складываем имена файлов в словарь по категориям:
        for i in cat_paths:
            self.alltexts_dict[i.split('/')[-2]] = [i + j for j in os.listdir(i) if re.search('\w',open(i+j,'r',encoding='utf-8').read()) is not None]
        self.alltexts_list = []
        for k in self.alltexts_dict:
            self.alltexts_list += self.alltexts_dict[k]
        self.vectorizer.fit(self.alltexts_list)
        self.count_vectorizer = sktext.CountVectorizer(stop_words = stopwords,vocabulary=self.vectorizer.vocabulary_)
        self.features = self.vectorizer.get_feature_names()
        self.alltext = get_one_text(self.alltexts_list)
        self.counts_alltext = np.array(self.count_vectorizer.transform([self.alltext]).mean(axis=0))[0]
        

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

    def keyword_chi_square(self, d1):
        '''на вход поступают уже готовые словари ключевых слов для
подкорпуса и всего корпуса'''
        idfs = tuple(self.vectorizer.idf_)
        indices = get_indices(d1, self.features, reverse = True)
        d2 = self.get_corpora_average(d1)
        d1 = {k:v*1000000/idfs[indices[k]] for k,v in d1.items()}
        d2 = {k:v*1000000/idfs[indices[k]] for k,v in d2.items()}
        return sum([((d1[k]-d2[k])**2)/d1[k] for k in d1])

    def keyword_chi_square_counts(self, keywords, category):
        chi = 0
        stat = text_len_stat(self.vectorizer.build_tokenizer())
        indices = get_indices(keywords, self.features)
        category_text = get_one_text(self.alltexts_dict[category])
        tokenize = self.count_vectorizer.build_tokenizer()
        category_len = len(tokenize(category_text))
        alltext_len = len(tokenize(self.alltext))
        counts_category = np.array(self.count_vectorizer.transform([category_text]).mean(axis=0))[0]
        counts_category = tuple(i for i in counts_category)
        counts_alltext = tuple(i*category_len/alltext_len for i in self.counts_alltext)
        for i in indices:
            chi += ((counts_category[i]-counts_alltext[i])**2)/counts_category[i]
        return chi

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
            l = len(self.tokenize(f))
            if l==0:
                print('Файл с 0 слов: ',t)
            lens.append(l)
        return {'Дисперсия':statistics.pvariance(lens),'Стандартное отклонение':statistics.pstdev(lens),'Максимум':max(lens),'Минимум':min(lens)}
        
     
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

def csv_parse(filename):
    outp = []
    with open (filename, 'r', encoding='utf-8') as f:
        x = f.readlines()
    cols = x[0].split('\t')
    for i in x[1:]:
        outp.append({j.strip():k.strip() for j,k in zip(cols,i.split('\t'))})
    all_pos = set(i['PoS'] for i in outp)
    print(all_pos)
    return outp

def get_one_text(files):
    t = ''
    for i in files:
        t += open(i,'r',encoding='utf-8').read()
    return t

##пример использования - можно закомментить
if __name__ == '__main__':
    a = mean_tfidf_by_cat(('./MPGU/','./MGTU/','./VShe/','./REU/','./MISiS/','./MGU/'))
    MPGU = a.get_keywords('MPGU',25)
    print('МПГУ')
    print(MPGU)
    print('chi-square:',a.keyword_chi_square_counts(MPGU,'MPGU'))
    MGTU = a.get_keywords('MGTU',25)
    print('МГТУ')
    print(MGTU)
    print('chi-square:',a.keyword_chi_square_counts(MGTU,'MGTU'))
    VShe = a.get_keywords('VShe',25)
    print('ВШЭ')
    print(VShe)
    print('chi-square:',a.keyword_chi_square_counts(VShe,'VShe'))
    REU = a.get_keywords('REU',25)
    print('РЭУ')
    print(REU)
    print('chi-square:',a.keyword_chi_square_counts(REU,'REU'))
    MISiS = a.get_keywords('MISiS',25)
    print('МИСиС')
    print(MISiS)
    print('chi-square:',a.keyword_chi_square_counts(MISiS,'MISiS'))
    MGU = a.get_keywords('MGU',25)
    print('МГУ')
    print(MGU)
    print('chi-square:',a.keyword_chi_square_counts(MGU,'MGU'))
    stat = text_len_stat(a.vectorizer.build_tokenizer())
    print('Длина текстов')
    print('Средняя:')
    print(stat.get_average_len(a.alltexts_list))
    print(stat.get_stats(a.alltexts_list))
