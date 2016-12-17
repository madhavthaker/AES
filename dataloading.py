from requiredimports import *

xl_workbook = pd.ExcelFile('training_set_rel3.xlsx')
df_all = xl_workbook.parse("training_set")
df_all = df_all[df_all['domain1_score']<61]
df_all = df_all.dropna(axis = 1)
df_all = df_all.drop('rater1_domain1', 1)
df_all = df_all.drop('rater2_domain1', 1)
df_all = df_all.drop('essay_id', 1)

essay_sets = np.unique(df_all['essay_set'])

for set_no in essay_sets:
    indices = df_all[df_all['essay_set'] == set_no].index.tolist()
    grade_max = np.max(df_all.loc[indices, 'domain1_score'])
    grade_min = np.min(df_all.loc[indices, 'domain1_score'])
    df_all.loc[indices, 'domain1_score_12'] = 12*(df_all.loc[indices, 'domain1_score'] - grade_min)/(grade_max - grade_min)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_all['essay'], df_all['domain1_score_12'], test_size=0.10)
#X_train, X_test, y_train, y_test = train_test_split(df_all['essay'], df_all['domain1_score'], test_size=0.10)

y_train = np.reshape(y_train,(len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

