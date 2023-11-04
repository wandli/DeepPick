import sklearn
# from sklearn import cross_validation   # 版本更新后修改为下一行
# from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import ensemble, svm
from xgboost import XGBClassifier, plot_importance, Booster
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import collections
from numpy.random import shuffle
from matplotlib import pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

header = './dataset/predict/'
locfile = header +'/loc2.csv'         # s + o  _static
t_sz = 0.2
data_s, data_o, data_l = 0, 0, 0


def mcnemar_test(table):
    result = mcnemar(table, exact=False)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

def eq(a, b):
    if a == b:
        return 1
    else:
        return 0


def get_mc(method):
    table = [[0,0],[0,0]]
    for index, row in df.iterrows():
        if row['cat'] == 1:
            if row[method] == 1:
                table[0][0] += 1
            else:
                table[0][1] += 1
        else:
            if row[method] == 0:
                table[1][1] += 1
            else:
                table[1][0] += 1
    return table


def test_right_predict(df):
    df['cat'] = df.apply(lambda x:eq(x[0], x[1]), axis=1)
    df['xgb'] = df.apply(lambda x:eq(x[0], x['XGBoost']), axis=1)
    df['cart'] = df.apply(lambda x:eq(x[0], x['CART']), axis=1)
    df['rf'] = df.apply(lambda x:eq(x[0], x['Random Forest']), axis=1)
    return df

# for yelp esize and betweenness
def get_non_strip_user(user_type):
    s = pd.read_csv(header + user_type+ '.txt', sep=' ')
    o = pd.read_csv(header + user_type+ 'ord.txt', sep=' ')
    return list(s['id']), list(o['id'])

# for yelp others (elite, const, hier)
def get_strip_user(user_type):
    with open(header + user_type+ '.txt') as f:
        s = f.read().strip().split('\n')
    with open(header + user_type+ 'ord.txt') as f:
        o = f.read().strip().split('\n')
    return s, o

def get_kcore():
    loc=header + 'k-shell'
    with open(loc+'/shell.txt', 'r', encoding='utf-8') as f:
        kcore = f.read().split('\n')
        kcore = [int(i) for i in kcore]
    with open(loc+'/shellord.txt', 'r', encoding='utf-8') as f:
        kcoreord = f.read().split('\n')
        kcoreord = [int(i.split(' ')[0]) for i in kcoreord]
    return kcore, kcoreord


# get user review time features
# for shs, identity=1; for ord, identity=0
# may need modify
def get_review(sepc, sepc_name="shell"):
    uf = pd.read_csv('./dataset/' + sepc_name + '_time_features.csv')
    features = uf.drop('review_frequency', axis=1)
    return features


# all read function sort user_id by number
# read LIWC data
def get_LIWC(shs, ordu):
    s = pd.read_csv('dataset/LIWC_id_avg.csv', encoding='utf-8')
    s1 = s[s['user_id'].isin(shs)]
    o1 = s[s['user_id'].isin(ordu)]
    conc = pd.concat([s1, o1], axis=0)
    conc = conc[['user_id', 'WC', 'Analytic', 'focuspast', 'social', 'leisure']] # 'shehe', 'pronoun', 'i',
    # conc = conc[['user_id', 'sexual', 'death', 'relig', 'swear', 'Colon', 'filler']]
    conc['attr'] = np.array([1] * len(s1) + [0] * len(o1))
    return conc


# read loc data
# now is all features
def get_LOC(shs, ordu):
    data_l = pd.read_csv(locfile, encoding='utf-8')
    s1 = data_l[data_l['user_id'].isin(shs)]
    o1 = data_l[data_l['user_id'].isin(ordu)]
    data_l = pd.concat([s1, o1], axis=0)
    # 10k rows in total
    # print(data_l.shape)
    data_l['attr'] = np.array([1] * len(s1) + [0] * len(o1))
    selected = data_l.loc[:, ['homo_min', 'en_max', 'homo_max',
                          'en_small_q', 'en_min', 'homo_large_q', 'user_id', 'attr']]
    return selected


# get all of user features
def get_user_feature(shs, ordu):
    # get selected users
    uf = pd.read_csv('dataset/user_feature.csv', engine='python').drop('review_count', axis=1)
    uf = uf.loc[:, ['user_id', 'yelping_since', 'funny', 'compliment_hot', 'compliment_cool', 'compliment_funny']] # 'attr', 
    uf['yelping_since'] = uf['yelping_since'] / 100000000
    
    s1 = uf[uf['user_id'].isin(shs)]
    o1 = uf[uf['user_id'].isin(ordu)]    
    
    uf = pd.concat([s1, o1], axis=0)
    uf['attr'] = np.array([1] * len(s1) + [0] * len(o1))
    return uf


# get deep learning predict results
def get_deep_learning(sepc):
    dl = pd.read_csv('result/' + sepc + 'review_seq_plstm.csv')#.drop('index', axis=1)
    return dl


def skip_non_float(val):
    ans = []
    for i in val:
        try:
            i = float(i)
            ans.append(i)
        except ValueError:
            print('error' + i + '.')
    return ans


# get txtcnn results
def get_txtcnn(sepc):
    spdf = {}
    ordf = {}
    with open('dataset/' + sepc + '_txtcnn.txt', 'r', encoding='utf-8') as fs:
        f = fs.read().strip('\n').split('\n')
        for line in f:
            line = line.split(',')
            val = line[1].split('|')
            if not line[0].isdigit():
                print(line)
            spdf[line[0]] = skip_non_float(val)
    with open('dataset/' + sepc + 'ord_txtcnn.txt', 'r', encoding='utf-8') as fs:
        f = fs.read().strip('\n').split('\n')
        for line in f:
            line = line.split(',')
            val = line[1].split('|')
            if not line[0].isdigit():
                print(line)
            ordf[line[0]] = skip_non_float(val)
    num_s = len(spdf)
    num_o = len(ordf)
    spdf = pd.DataFrame.from_dict(spdf, orient='index')
    ordf = pd.DataFrame.from_dict(ordf, orient='index')
    result = pd.concat([spdf, ordf], axis=0)
    result.columns = [str(i) for i in result.columns]
    result['attr'] = [1] * num_s + [0] * num_o
    result['user_id'] = np.array(result.index, dtype=int)
    # print(result.index)
    # print(num_s + num_o)
    # for x in result.index:
    #     if not x.isdigit():
    #         print(x)
    return result


# get crnn results
def get_crnn(sepc_name):
    spdf = {}
    ordf = {}
    with open('dataset/' + sepc_name + '_CRNN.txt', 'r', encoding='utf-8') as fs:
        f = fs.read().strip('\n').split('\n')
        for line in f:
            line = line.split(',')
            val = line[1].split('|')
            if not line[0].isdigit():
                print(line)
            spdf[line[0]] = skip_non_float(val)
    with open('dataset/' + sepc_name + 'ord_CRNN.txt', 'r', encoding='utf-8') as fs:
        f = fs.read().strip('\n').split('\n')
        for line in f:
            line = line.split(',')
            val = line[1].split('|')
            if not line[0].isdigit():
                print(line)
            ordf[line[0]] = skip_non_float(val)
    num_s = len(spdf)
    num_o = len(ordf)
    spdf = pd.DataFrame.from_dict(spdf, orient='index')
    ordf = pd.DataFrame.from_dict(ordf, orient='index')
    result = pd.concat([spdf, ordf], axis=0)
    result.columns = [str(i) for i in result.columns]
    result['attr'] = [1] * num_s + [0] * num_o
    result['user_id'] = np.array(result.index, dtype=int)
    return result


# get a list of user features, all includes their ids and attributes
# contact them together by index (id) and separate training and testing set
def cont_features(flst, sepc):
    llst = len(flst)
    totfea = flst[0]
    if llst > 1:
        for i in range(1, llst):
            if 'attr' in totfea and 'attr' in flst[i]:
                totfea = pd.merge(totfea, flst[i], how='left',
                          left_on=['user_id', 'attr'], right_on=['user_id', 'attr'])
            else:
                totfea = pd.merge(totfea, flst[i], how='left', left_on=['user_id'], right_on=['user_id'])
            if not totfea.columns[np.where(np.isnan(totfea))[1]].empty:
                print('Produced NAN, feature id: {0}'.format(i))
    totfea = totfea.set_index('user_id')
    # 如果含有nan
    if not totfea.columns[np.where(np.isnan(totfea))[1]].empty:
        totfea.dropna(axis=0, how='any',inplace=True)
    print('remain shape of {0} users is: {1}'.format(sepc, str(totfea.shape)))
    return totfea    # .loc[:,['attr', 'review_count']]


def draw_cor_matrx(totfea):
    df = totfea.drop('attr', axis=1)
    f = plt.figure(figsize=(30, 25))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


# select feature by chi2
# returning a pair of arrays (scores, pvalues) or a single array with scores.
def plot_chi2_scores(x, y, filename):
    names = list(x.columns)
    # 1. 使用 sklearn.feature_selection.SelectKBest 给特征打分
    slct = SelectKBest(chi2, k="all")
    x[x < 0] = 100000
    slct.fit(x, y)
    scores = slct.scores_

    # 2. 将特征按分数 从大到小 排序
    named_scores = zip(names, scores)
    sorted_named_scores = sorted(named_scores, key=lambda z: z[1], reverse=True)

    sorted_scores = [each[1] for each in sorted_named_scores]
    sorted_names = [each[0] for each in sorted_named_scores]

    y_pos = np.arange(len(names))  # 从上而下的绘图顺序

    # 3. 绘图
    fig, ax = plt.subplots()
    ax.barh(y_pos, sorted_scores, height=0.7, align='center', color='#AAAAAA', tick_label=sorted_names)
    # ax.set_yticklabels(sorted_names)      # 也可以在这里设置 条 的标签
    ax.set_yticks(y_pos)
    ax.set_xlabel('Feature Score')
    ax.set_ylabel('Feature Name')
    ax.invert_yaxis()
    ax.set_title('chi2 scores of user profile features')

    # 4. 添加每个bar的数字标签
    for score, pos in zip(sorted_scores, y_pos):
        ax.text(score + 2000, pos, '%.1f' % score, ha='center', va='bottom', fontsize=8)
    # fig.set_size_inches(18.5, 10.5)
    fig.set_size_inches(25, 12)
    fig.savefig(filename + ".png", bbox_inches='tight')
    plt.show()


# find best para in XGB
def xgb_find_para():
    # n_estimators para
    # cv_params = {'n_estimators': [10, 100, 200, 300, 400]}
    # cv_params = {'n_estimators': [900, 800, 1000, 1200, 1400]}
    # cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    # min_child_weight & max_depth para
    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}    # max: 3, min: 1
    # gamma para
    # cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}    # 0.3 FOR profile
    # subsample and colsample_bytree
    cv_params = {'subsample': [0.6, 0.5, 0.4, 0.3, 0.2], 'colsample_bytree': [0.6, 0.5, 0.4, 0.3, 0.2], 'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5]}
    other_params = {'learning_rate': 0.01, 'n_estimators': 900, 'max_depth': 3, 'min_child_weight': 1}

    clf = XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=clf, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=5)
    return optimized_GBM


def cat_find_para():
    cat = CatBoostClassifier(loss_function='Logloss')
    para = {'iterations':[10, 100, 500, 1000],
            'learning_rate':[0.01, 0.1, 1],
            'depth':[2, 4, 7, 10],
            'l2_leaf_reg' : [1, 4, 9],
            # 'border_count': 122,
            'one_hot_max_size' : [3, 8]
    }
    clf = GridSearchCV(estimator=cat, param_grid=para, scoring='roc_auc', cv=5, verbose=1, n_jobs=5)
    return clf


# grid search the best para
def find_para(optimized_GBM, X_train, X_test, y_train, y_test):
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    test_pred = optimized_GBM.predict(X_test)
    print('auc: ' + str(sklearn.metrics.roc_auc_score(y_test, test_pred)))
    print('f1_score: ' + str(sklearn.metrics.f1_score(y_test, test_pred)))


def classifier(X_train, X_test, y_train, y_test, cls, importance_type='gain'):
    
    if cls == 'xgb':
        model = XGBClassifier(learning_rate=0.01,
                                  n_estimators=900,
                                  max_depth=3,
                                  min_child_weight=1, seed=0,
                                  subsample=0.6,
                                  colsample_bytree=0.6,
                                  gamma=0.1,
                                  reg_alpha=1, 
                                  reg_lambda=1,
                                  n_jobs=5)
    elif cls == 'ctb':
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=4,
            l2_leaf_reg=9,
            one_hot_max_size=3,
            loss_function='Logloss')
    elif cls == 'rf':
        model = ensemble.RandomForestClassifier(max_depth=7,
                                                n_estimators=130,
                                                n_jobs=5)
    elif cls == 'dt':
        model = sklearn.tree.DecisionTreeClassifier()
    elif cls == 'svm':
        grid = GridSearchCV(svm.SVC(), param_grid={"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01,0.001, 0.0001]}, cv=5)
        grid.fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        model = svm.SVC(kernel='rbf', C=400, gamma='auto')  # , gamma='auto'
    elif cls == 'lr':
        model = LogisticRegression(n_jobs=5)
    else:
        print('have not specified model. Exiting..')
        return
    model.fit(X_train, y_train)
#     if cls == 'xgb':
#         fig, ax = plt.subplots(figsize=(25, 20))
#         plt.sca(ax=ax)
#         plot_importance(model, importance_type)
#         plt.show()
    '''
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print('scores: ' + str(scores))
    print('scores mean: ' + str(scores.mean()))
    if cls == 'ctb':
        model.save_model('{0}_{1}_model.h5'.format(sepc, method))
        importance = model.get_feature_importance()
        print(importance)
        print(list(X_train.columns))
        np.savetxt(sepc + "importance.csv", importance, delimiter=",")
        test_vakye=0
    '''
    return eval(X_test, y_test, model), model


def plot_imp(model, importance_type='gain'):
    importance = model.get_feature_importance()
    imp = model.get_booster().get_score(importance_type=importance_type)
    imp2 = collections.OrderedDict(sorted(imp.items(), key=lambda x:x[1]))
    dfi = pd.DataFrame.from_dict(imp2, orient='index')
    dfi.to_csv('importance/' + sp + '_xgb_' + importance_type + '.csv')
    print(dfi)   


def eval(X_test, y_test, model):
    test_pred = model.predict(X_test)
    # cm_train = sklearn.metrics.confusion_matrix(y_train, model.predict(X_train))  # 训练集混淆矩阵
    # cm_test = sklearn.metrics.confusion_matrix(y_test, pred)  # 测试集混淆矩阵
    acc = sklearn.metrics.accuracy_score(y_test, test_pred)
    precision_score = sklearn.metrics.precision_score(y_test, test_pred)
    recall_score = sklearn.metrics.recall_score(y_test, test_pred)
    f1_score = sklearn.metrics.f1_score(y_test, test_pred)
    auc = sklearn.metrics.roc_auc_score(y_test, test_pred)
    print('acc: ' + str(acc))
    print('precision_score: ' + str(precision_score))
    print('recall_score: ' + str(recall_score))
    print('f1_score: ' + str(f1_score))
    print('auc: ' + str(auc))
    return [acc, precision_score, recall_score, f1_score, auc]


# 读取各类feature信息
def get_all_features(nntype, shs, ord, only_nn=False, sepc_name="shell"):
    if not only_nn:
        # get LOC features
        loc = get_LOC(shs, ord)

        # get LIWC features
        liwc = get_LIWC(shs, ord)

        # get best user profile feature
        uf = get_user_feature(shs, ord)

        # get review features
        revf = get_review(shs, sepc_name)
    
    dlf = pd.DataFrame()
    nodlf = False
    if nntype == 'plstm':
        # get deep learning result
        dlf = get_deep_learning(sepc_name)
    elif nntype == 'textcnn' or nntype == 'txtcnn':
        # get txtcnn result
        dlf = get_txtcnn(sepc_name)
    elif nntype == 'crnn':
        # get CRNN result
        dlf = get_crnn(sepc_name)
    else:
        nodlf = True
        
    if only_nn:
        return [dlf]
    if nodlf:
        return [loc, liwc, revf, uf]
    return [loc, liwc, revf, uf, dlf]  #


# original main
def get_all_ml():
    sepc = 'shell'
    method = 'crnn'  #

    if sepc in ['elite', 'const', 'hier']:
        shs, ordu = get_strip_user(sepc)
    elif sepc in ['betw', 'esize']:
        shs, ordu = get_non_strip_user(sepc)
    elif sepc == 'shell':
        shs, ordu = get_kcore()
    else:
        print('wrong input. user should be classified as elite, betw or shell.')

    # contact all features
    features = get_all_features(method, shs, ordu, sepc_name=sepc)

    totfea = cont_features(features, sepc)

    # split test and train
    X_train, X_test, y_train, y_test = train_test_split(totfea.drop('attr', axis=1), totfea['attr'],
                                                                       test_size=t_sz, random_state=1)

    df = pd.DataFrame()
    print('xgboost: ')
    xgb, modelb = classifier(X_train, X_test, y_train, y_test, cls='xgb')
    modelb.get_booster().get_score(importance_type='weight')
    df['XGBoost'] = xgb
    print('catboost: ')
    ctb, modelc = classifier(X_train, X_test, y_train, y_test, cls='ctb')
    df['CatBoost'] = ctb

    print('decision tree: ')
    dt, modeld = classifier(X_train, X_test, y_train, y_test, cls='dt')
    df['CART'] = dt

    print('random forest: ')
    rf, modelr = classifier(X_train, X_test, y_train, y_test, cls='rf')
    df['Random Forest'] = rf

    df.to_csv('result/' + sepc + '_' + method + '.csv')


if __name__ == "__main__":
    get_all_ml()