import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('utils/')
from utils import print_time

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)                    
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print('dtypes: ', df.dtypes)
    return df

def preprocess_avazu(data_path, feature_value_filter=False, threshold=4):
    print('start reading file...')
    df_train = pd.read_csv(data_path + 'train.csv')
    df_val = pd.read_csv(data_path + 'valid.csv')
    df_test = pd.read_csv(data_path + 'test.csv')
    df = pd.concat([df_train, df_val, df_test])
    del df_train, df_val, df_test
    print('finish reading file...')
    df.drop(columns=['id'], inplace=True)
    # transform hour to hour
    # df['hour:token'] = pd.to_datetime(df['timestamp:float'], format='%y%m%d%H')
    # df['hour:token'] = df['hour:token'].dt.hour
    # df.drop(['timestamp:float'], axis=1, inplace=True)
    sparse_features = [f for f in df.columns]
    # df = df.fillna('-1')

    if feature_value_filter:
        print('start replace values')
        tqdm.pandas(desc='pandas bar')
        def replace_values(series):
            counts = series.value_counts()
            return series.apply(lambda x: -99 if counts[x] < threshold else x)
        df = df.parallel_apply(replace_values)
        print('finish replace values')
    df = df.astype(str)

    tk0 = tqdm(sparse_features, desc='LabelEncoder')
    for feat in tk0:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    df = df.infer_objects()
    df = reduce_mem_usage(df)
    df.to_csv(data_path + 'preprocessed_avazu.csv', index=False)

def preprocess_criteo(data_path, feature_value_filter=False, threshold=4):
    print_time('start reading file...')
#     df = pd.read_csv(data_path + 'criteo.inter', sep='\t')
    df = pd.read_csv(data_path + 'train.txt', sep='\t', header=None)
    print(df)
    print_time('finish reading file...')
    '''
    Index([ (label)0,  
    (float)1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
    (object)14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
      dtype='int64')
    '''
    df.columns= [str(x) for x in list(range(40))]
    dense_features = [f for f in df.columns.tolist() if (df[f].dtype in ['int64', 'float64'] and f != '0')]
    sparse_features = [f for f in df.columns.tolist() if df[f].dtype in ['object']]
    
    print_time('fill nan...')
    df[sparse_features] = df[sparse_features].fillna('-999')
    df[dense_features] = df[dense_features].fillna(-999)
    
    print_time('convert float features...')
    import math
    for feat in dense_features:
        df[feat] = df[feat].apply(lambda x:str(int(math.log(x) ** 2)) if x > 2 else str(int(x)-2))
    all_features = [f for f in df.columns]
    
#     df = df.astype(str)
    print_time('label encoding...')
    tk0 = tqdm(all_features, desc='LabelEncoder')
    for feat in tk0:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    df = df.infer_objects()
    # 设置display.max_rows选项
    pd.set_option('display.max_rows', None)
    df = reduce_mem_usage(df)
    
    print_time('save to file...')
    df.to_csv(data_path + 'preprocessed_criteo.csv', index=False)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avazu', help='avazu, criteo')
    parser.add_argument('--data_path', type=str, default='data/', help='data path')
    
    args = parser.parse_args()

    if args.dataset == 'avazu':
        preprocess_avazu(args.data_path + args.dataset + '/')
        print('preprocess avazu done!')
    elif args.dataset == 'criteo':
        preprocess_criteo(args.data_path + args.dataset + '/')
        print('preprocess criteo done!')