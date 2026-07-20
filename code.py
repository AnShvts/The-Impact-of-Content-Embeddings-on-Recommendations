# Установка библиотек
!pip install implicit scikit-learn polars pyarrow huggingface_hub -q


# Импорт библиотек и модулей
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.bpr import BayesianPersonalizedRanking
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from huggingface_hub import hf_hub_download
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')


# Создание папок на диске
drive.mount('/content/drive')


os.makedirs('/content/drive/MyDrive/VK_LSVD_Results/plots', exist_ok=True)
os.makedirs('/content/drive/MyDrive/VK_LSVD_Results/models', exist_ok=True)
os.makedirs('/content/drive/MyDrive/VK_LSVD_Results/reports', exist_ok=True)


# Скачивание данных из датасета
subsample = 'up0.001_ip0.001'
local_dir = '/content/VK-LSVD'


train_files = [f'subsamples/{subsample}/train/week_{i:02}.parquet' for i in range(25)]
val_file = f'subsamples/{subsample}/validation/week_25.parquet'
metadata_files = [
    'metadata/users_metadata.parquet',
    'metadata/items_metadata.parquet',
    'metadata/item_embeddings.npz'
]
all_files = train_files + [val_file] + metadata_files


for file in tqdm(all_files, desc='Загрузка'):
    hf_hub_download(
        repo_id='deepvk/VK-LSVD',
        repo_type='dataset',
        filename=file,
        local_dir=local_dir
    )


# Загрузка метаданных
users_meta = pl.read_parquet(f'{local_dir}/metadata/users_metadata.parquet')
items_meta = pl.read_parquet(f'{local_dir}/metadata/items_metadata.parquet')


# Загрузка эмбеддингов
embeddings_npz = np.load(f'{local_dir}/metadata/item_embeddings.npz')
item_ids_all = embeddings_npz['item_id']
item_embs_all = embeddings_npz['embedding']


# Загрузка взаимодействий
train_dfs = []
for i in range(25):
    file = f'{local_dir}/subsamples/{subsample}/train/week_{i:02}.parquet'
    df = pl.read_parquet(file, columns=['user_id', 'item_id'])
    train_dfs.append(df)
train_df = pl.concat(train_dfs)
del train_dfs


# Валидация
val_df = pl.read_parquet(f'{local_dir}/subsamples/{subsample}/validation/week_25.parquet',
                         columns=['user_id', 'item_id'])


# Фильтрация
user_counts = train_df.group_by('user_id').agg(pl.count().alias('n'))
user_counts = user_counts.filter(pl.col('n') >= 5)
active_users = user_counts['user_id'].to_list()


item_counts = train_df.group_by('item_id').agg(pl.count().alias('n'))
item_counts = item_counts.filter(pl.col('n') >= 3)
popular_items = item_counts['item_id'].to_list()


train_df = train_df.filter(pl.col('user_id').is_in(active_users))
train_df = train_df.filter(pl.col('item_id').is_in(popular_items))


# Построение матрицы
users = train_df['user_id'].unique().to_numpy()
items = train_df['item_id'].unique().to_numpy()


user_to_idx = {u: i for i, u in enumerate(users)}
item_to_idx = {i: j for j, i in enumerate(items)}
idx_to_item = {j: i for i, j in item_to_idx.items()}


rows = [user_to_idx[u] for u in train_df['user_id']]
cols = [item_to_idx[i] for i in train_df['item_id']]
data = np.ones(len(rows), dtype=np.float32)


matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))


# Фильтрация эмбеддингов
mask = np.isin(item_ids_all, items)
filtered_item_ids = item_ids_all[mask]
filtered_embs = item_embs_all[mask]


# Сортировка
item_order = np.argsort(items)
filtered_embs = filtered_embs[item_order]


# Фильтрация валидации
val_df = val_df.filter(pl.col('user_id').is_in(users))
val_df = val_df.filter(pl.col('item_id').is_in(items))


# Построение валидационной матрицы
val_rows = [user_to_idx[u] for u in val_df['user_id']]
val_cols = [item_to_idx[i] for i in val_df['item_id']]
val_data = np.ones(len(val_rows), dtype=np.float32)


val_matrix = csr_matrix((val_data, (val_rows, val_cols)), shape=(len(users), len(items)))


# Сохранение данных на диск
save_npz('/content/drive/MyDrive/VK_LSVD_Results/interaction_matrix.npz', matrix)
save_npz('/content/drive/MyDrive/VK_LSVD_Results/validation_matrix.npz', val_matrix)


with open('/content/drive/MyDrive/VK_LSVD_Results/mappings.pkl', 'wb') as f:
    pickle.dump({
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'users': users,
        'items': items
    }, f)


np.save('/content/drive/MyDrive/VK_LSVD_Results/item_embeddings.npy', filtered_embs)


# Реализация контентной модели на основе эмбеддингов
class ContentOnlyRecommender:


    def __init__(self, item_embeddings):


        self.item_embeddings = item_embeddings


    def recommend(self, user_idx, matrix, N=10):


        user_items = matrix[user_idx].nonzero()[1]
        if len(user_items) == 0:
            indices = np.random.choice(len(self.item_embeddings), min(N, len(self.item_embeddings)), replace=False)
            return [(i, 0.0) for i in indices]


        user_emb = np.mean(self.item_embeddings[user_items], axis=0)
        scores = []
        for i, emb in enumerate(self.item_embeddings):
            if i not in user_items:
                norm_u = np.linalg.norm(user_emb)
                norm_i = np.linalg.norm(emb)
                if norm_u > 0 and norm_i > 0:
                    sim = np.dot(user_emb, emb) / (norm_u * norm_i)
                else:
                    sim = 0
                scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:N]


# Модель Popularity
class PopularityRecommender:
    def __init__(self, matrix):
        self.popularity = np.array(matrix.sum(axis=0)).flatten()


    def recommend(self, user_idx, matrix, N=10):
        user_items = set(matrix[user_idx].nonzero()[1])
        candidates = [(i, self.popularity[i]) for i in range(len(self.popularity)) if i not in user_items]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in candidates[:N]]


# Функция для построения матрицы косинусного сходства
def build_similarity_matrix(item_embeddings, top_n=5000):


    item_counts = np.array(matrix.sum(axis=0)).flatten()    # Топ-N популярных видео
    top_indices = np.argsort(item_counts)[-top_n:]


    embeddings_subset = item_embeddings[top_indices]    # Эмбеддинги для этих видео


    item_sim = cosine_similarity(embeddings_subset)    # Подсчёт косинусного сходства


    idx_to_sub_idx = {full_idx: sub_idx for sub_idx, full_idx in enumerate(top_indices)}    # Маппинг индексов
    sub_idx_to_idx = {sub_idx: full_idx for full_idx, sub_idx in idx_to_sub_idx.items()}


    return item_sim, idx_to_sub_idx, sub_idx_to_idx


# Реализация гибридного подхода: ALS + контентное сходство
def hybrid_recommend(user_idx, model, item_sim, idx_to_sub_idx, train_matrix, k=10, alpha=0.5):


    try:
        user_items = train_matrix[user_idx:user_idx+1]    # ALS рекомендации
        als_recs = model.recommend(user_idx, user_items, N=k*2, filter_already_liked_items=True)


        if len(als_recs) == 0:
            return np.array([])


        als_items = np.array([int(rec[0]) for rec in als_recs])
        als_scores = np.array([rec[1] for rec in als_recs])


        hybrid_scores = []
        for i, item in enumerate(als_items):
            if item in idx_to_sub_idx:    # Контентная часть
                sub_idx = idx_to_sub_idx[item]
                sim_score = np.mean(item_sim[sub_idx])
            else:
                sim_score = 0


            hybrid_score = alpha * als_scores[i] + (1 - alpha) * sim_score    # Взвешенная сумма
            hybrid_scores.append(hybrid_score)


        sorted_indices = np.argsort(-np.array(hybrid_scores))    # Сортировка
        return als_items[sorted_indices[:k]]


    except Exception as e:
        return np.array([])


# Функция оценки ALS
def evaluate_als(model, matrix, val_matrix, idx_to_item, user_to_idx, k=10, n_users=300):


    val_users = val_matrix.nonzero()[0]
    if len(val_users) == 0:
        return {'precision@10': 0.0, 'recall@10': 0.0, 'ndcg@10': 0.0}


    sample_users = np.random.choice(val_users, size=min(n_users, len(val_users)), replace=False)


    precisions = []
    recalls = []
    ndcgs = []
    success_count = 0


    for user_idx in tqdm(sample_users, desc='  Оценка ALS'):
        relevant_indices = val_matrix[user_idx].nonzero()[1]    # Получение релевантных видео из валидации
        if isinstance(relevant_indices, tuple):
            relevant_indices = relevant_indices[0]
        relevant_indices = np.array(relevant_indices).flatten()


        if len(relevant_indices) == 0:
            continue


        try:
            from scipy.sparse import csr_matrix   # Использование user_items как csr_matrix с одной строкой
            user_history = matrix[user_idx].nonzero()[1]


            if len(user_history) == 0:
                continue


            user_items = csr_matrix(([1.0] * len(user_history), ([0] * len(user_history), user_history)),
                                    shape=(1, matrix.shape[1]))


            recs = model.recommend(user_idx, user_items, N=k, filter_already_liked_items=True)
            recommended = [int(rec[0]) for rec in recs]


        except Exception as e:
            continue


        if len(recommended) == 0:
            continue


        success_count += 1


        relevant_set = set(relevant_indices)    # Подсчёт метрик
        rec_set = set(recommended[:k])


        precision = len(relevant_set & rec_set) / k
        precisions.append(precision)


        recall = len(relevant_set & rec_set) / min(k, len(relevant_indices))
        recalls.append(recall)


        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)


        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_indices))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)


    return {
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'ndcg@10': np.mean(ndcgs) if ndcgs else 0.0
    }


# Функция оценки гибридной модели
def calc_hybrid_metrics(model, item_sim, idx_to_sub_idx, train_matrix, val_matrix, k=10, alpha=0.5, n_users=300):


    precisions = []
    recalls = []
    ndcgs = []
    success_count = 0


    val_users = val_matrix.nonzero()[0]    # Случайные пользователи из валидации
    if len(val_users) == 0:
        return {'precision@10': 0.0, 'recall@10': 0.0, 'ndcg@10': 0.0}


    sample_users = np.random.choice(val_users, size=min(n_users, len(val_users)), replace=False)


    for user_idx in tqdm(sample_users, desc='  Оценка'):
        relevant_indices = val_matrix[user_idx].nonzero()[1]    # Реальные видео пользователя в валидации
        if len(relevant_indices) == 0:
            continue


        recommended = hybrid_recommend(user_idx, model, item_sim, idx_to_sub_idx, train_matrix, k, alpha)    # Получение рекомендаций


        if len(recommended) == 0:
            continue


        success_count += 1
        relevant_set = set(relevant_indices)
        rec_set = set(recommended[:k])


        precision = len(relevant_set & rec_set) / k    # Precision@k
        precisions.append(precision)


        recall = len(relevant_set & rec_set) / min(k, len(relevant_indices))    # Recall@k
        recalls.append(recall)


        dcg = 0.0    # NDCG@k
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)


        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_indices))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)


    return {
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'ndcg@10': np.mean(ndcgs) if ndcgs else 0.0
    }


# Функция оценки ItemKNN с проверкой индексов
def evaluate_knn(model, matrix, val_matrix, idx_to_item, user_to_idx, k=10, n_users=300):


    val_users = val_matrix.nonzero()[0]
    if len(val_users) == 0:
        return {'precision@10': 0.0, 'recall@10': 0.0, 'ndcg@10': 0.0}


    sample_users = np.random.choice(val_users, size=min(n_users, len(val_users)), replace=False)


    precisions = []
    recalls = []
    ndcgs = []
    success_count = 0


    for user_idx in tqdm(sample_users, desc='  Оценка KNN'):
        relevant_indices = val_matrix[user_idx].nonzero()[1]
        if isinstance(relevant_indices, tuple):
            relevant_indices = relevant_indices[0]
        relevant_indices = np.array(relevant_indices).flatten()


        if len(relevant_indices) == 0:
            continue


        try:    # Получение истории пользователя
            user_history = matrix[user_idx].nonzero()[1]
            if len(user_history) == 0:
                continue


            from scipy.sparse import csr_matrix    # Создание однострочной матрицы
            user_items = csr_matrix(([1.0] * len(user_history), ([0] * len(user_history), user_history)),
                                    shape=(1, matrix.shape[1]))


            recs = model.recommend(user_idx, user_items, N=min(100, matrix.shape[1]),    # Получение рекомендаций
                                  filter_already_liked_items=True)


            recommended = []    # Фильтрация валидных индексов
            for rec in recs:
                item_idx = int(rec[0])
                if item_idx in idx_to_item:
                    recommended.append(item_idx)


        except Exception as e:
            continue


        if len(recommended) == 0:
            continue


        success_count += 1


        relevant_set = set(relevant_indices)    # Подсчёт метрик
        rec_set = set(recommended[:k])


        precision = len(relevant_set & rec_set) / k
        precisions.append(precision)


        recall = len(relevant_set & rec_set) / min(k, len(relevant_indices))
        recalls.append(recall)


        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)


        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_indices))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)


    return {
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'ndcg@10': np.mean(ndcgs) if ndcgs else 0.0
    }


# Функция оценки моделей Content-Only, Popularity и Random
def evaluate_model_else(model, matrix, val_matrix, idx_to_item, user_to_idx, k=10, n_users=200):


    val_users = val_matrix.nonzero()[0]    # Получение пользователей из валидации
    if len(val_users) == 0:
        return {'precision@10': 0.0, 'recall@10': 0.0, 'ndcg@10': 0.0}


    sample_users = np.random.choice(val_users, size=min(n_users, len(val_users)), replace=False)    # Выборка пользователей


    precisions = []
    recalls = []
    ndcgs = []
    success_count = 0


    for user_idx in tqdm(sample_users, desc='  Оценка'):    # Получение индексов релевантных видео из валидации
        relevant_indices = val_matrix[user_idx].indices
        if len(relevant_indices) == 0:
            continue


        try:    # Получение рекомендаций
            if hasattr(model, 'recommend'):
                recs = model.recommend(user_idx, matrix, N=k)
                if recs and isinstance(recs[0], tuple):
                    recommended = [item for item, _ in recs]
                else:
                    recommended = recs
            else:
                continue
        except Exception as e:
            continue


        if len(recommended) == 0:
            continue


        success_count += 1


        relevant_set = set(relevant_indices)    # Подсчёт метрик
        rec_set = set(recommended[:k])


        precision = len(relevant_set & rec_set) / k
        precisions.append(precision)


        recall = len(relevant_set & rec_set) / min(k, len(relevant_indices))
        recalls.append(recall)


        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)


        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_indices))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)


    return {
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'ndcg@10': np.mean(ndcgs) if ndcgs else 0.0
    }


# Обучение ALS
als = []


for factors in [32, 64, 100]:   # Перебор параметров
    for reg in [0.01, 0.1, 0.5]:


        start = time.time()   # Обучение модели
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=reg,
            iterations=15,
            use_gpu=False,
            random_state=42
        )
        model.fit(matrix)


        metrics = evaluate_als(model, matrix, val_matrix, idx_to_item, user_to_idx, n_users=200)    # Оценка


        als.append({
            'factors': factors,
            'regularization': reg,
            'recall@10': metrics['recall@10'],
            'precision@10': metrics['precision@10'],
            'ndcg@10': metrics['ndcg@10']
        })


        print(f'P@10={metrics['precision@10']:.4f}, R@10={metrics['recall@10']:.4f}')


als_df = pd.DataFrame(als)
als_df = als_df.sort_values('recall@10', ascending=False)


print(als_df.to_string(index=False))


best_als = als_df.iloc[0]
print(f'\nЛучшая модель ALS: factors={best_als['factors']}, reg={best_als['regularization']}')
print(f'Recall@10: {best_als['recall@10']:.4f}')


als_df.to_csv('/content/drive/MyDrive/VK_LSVD_Results/reports/als_experiments_final.csv', index=False)


# Обучение гибридной модели
item_sim, idx_to_sub_idx, sub_idx_to_idx = build_similarity_matrix(filtered_embs, top_n=5000)    # Построение матрицы сходства


results = []


for factors in [50, 100]:
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        print(f'\nfactors={factors}, alpha={alpha}')


        start = time.time()    # Обучение модели
        model = AlternatingLeastSquares(
            factors=factors,
            iterations=20,
            regularization=0.1,
            use_gpu=False,
            random_state=42
        )
        model.fit(matrix)


        metrics = calc_hybrid_metrics(
            model, item_sim, idx_to_sub_idx,
            matrix, val_matrix,
            k=10, alpha=alpha, n_users=300
        )


        results.append({
            'factors': factors,
            'alpha': alpha,
            'precision@10': metrics['precision@10'],
            'recall@10': metrics['recall@10'],
            'ndcg@10': metrics['ndcg@10']
        })


        print(f'P@10={metrics['precision@10']:.4f}, R@10={metrics['recall@10']:.4f}')


results_df = pd.DataFrame(results)


# Лучшие показатели гибридной модели
best = results_df.loc[results_df['recall@10'].idxmax()]
print(f'\nЛучшие показатели гибридной модели')
print(f'factors={best['factors']}, alpha={best['alpha']}')
print(f'Recall@10: {best['recall@10']:.4f}')
print(f'Precision@10: {best['precision@10']:.4f}')
print(f'NDCG@10: {best['ndcg@10']:.4f}')


# Обучение и оценка модели ItemKNN
matrix_double = matrix.astype(np.float64)    # Обучение
knn_model = ItemItemRecommender(K=50)
knn_model.fit(matrix_double)


knn_metrics = evaluate_knn(knn_model, matrix_double, val_matrix, idx_to_item, user_to_idx, n_users=200)    # Оценка


print(f'\nItemKNN метрики:')
print(f'Recall@10: {knn_metrics['recall@10']:.4f}')
print(f'Precision@10: {knn_metrics['precision@10']:.4f}')
print(f'NDCG@10: {knn_metrics['ndcg@10']:.4f}')


# Модель Content-Only
content_results = []


for dim in [16, 32, 64]:
    print(f'\nРазмерность эмбеддингов: {dim}')


    embs_subset = filtered_embs[:, :dim]        # Первые dim компонент


    model = ContentOnlyRecommender(embs_subset)
    metrics = evaluate_model_else(model, matrix, val_matrix, idx_to_item, user_to_idx, n_users=200)


    content_results.append({
        'dimension': dim,
        'recall@10': metrics['recall@10'],
        'precision@10': metrics['precision@10'],
        'ndcg@10': metrics['ndcg@10']
    })


    print(f'Recall@10: {metrics['recall@10']:.4f}')
    print(f'Precision@10: {metrics['precision@10']:.4f}')
    print(f'NDCG@10: {metrics['ndcg@10']:.4f}')


content_df = pd.DataFrame(content_results)


best_content = content_df.loc[content_df['recall@10'].idxmax()]
print(f'\nЛучшая размерность для Content-Only: {best_content['dimension']}')
print(f'Recall@10: {best_content['recall@10']:.4f}')
print(f'Precision@10: {besct_ontent['precision@10']:.4f}')
print(f'NDCG@10: {best_content['ndcg@10']:.4f}')


content_df.to_csv('/content/drive/MyDrive/VK_LSVD_Results/reports/content_experiments.csv', index=False)


# Popularity
popular_model = PopularityRecommender(matrix)
popular_metrics = evaluate_model_else(popular_model, matrix, val_matrix, idx_to_item, user_to_idx, n_users=200)


print('\nPopularity')
print(f'Recall@10: {popular_metrics['recall@10']:.4f}')
print(f'Precision@10: {popular_metrics['precision@10']:.4f}')
print(f'NDCG@10: {popular_metrics['ndcg@10']:.4f}')


# Построение графиков ALS
csv_path = '/content/drive/MyDrive/VK_LSVD_Results/reports/als_experiments_final.csv'    # Путь к файлу
save_path = '/content/drive/MyDrive/VK_LSVD_Results/plots/'
os.makedirs(save_path, exist_ok=True)


als_df = pd.read_csv(csv_path)    # Загрузка данных
print(als_df.to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))    # Построение графиков


ax1 = axes[0]    # График 1: влияние regularization при разных factors
for factors in [32, 64, 100]:
    subset = als_df[als_df['factors'] == factors]
    ax1.plot(subset['regularization'], subset['recall@10'],
             marker='o', label=f'factors={factors}', linewidth=2, markersize=8)
ax1.set_xlabel('Regularization', fontsize=12)
ax1.set_ylabel('Recall@10', fontsize=12)
ax1.set_title('Влияние регуляризации на ALS', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)


for factors in [32, 64, 100]:    # Добавление значений на график
    subset = als_df[als_df['factors'] == factors]
    for _, row in subset.iterrows():
        ax1.text(row['regularization'], row['recall@10'] + 0.0001,
                f'{row["recall@10"]:.4f}', ha='center', va='bottom', fontsize=8)


ax2 = axes[1]    # График 2: тепловая карта
pivot_als = als_df.pivot(index='factors', columns='regularization', values='recall@10')
im = ax2.imshow(pivot_als.values, cmap='Blues', aspect='auto')
ax2.set_xticks(range(len(pivot_als.columns)))
ax2.set_yticks(range(len(pivot_als.index)))
ax2.set_xticklabels(pivot_als.columns)
ax2.set_yticklabels(pivot_als.index)
ax2.set_xlabel('Regularization', fontsize=12)
ax2.set_ylabel('Factors', fontsize=12)
ax2.set_title('Тепловая карта ALS (Recall@10)', fontsize=14)


for i in range(len(pivot_als.index)):    # Добавление значений в ячейки тепловой карты
    for j in range(len(pivot_als.columns)):
        value = pivot_als.iloc[i, j]
        ax2.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=9, color='black')


plt.colorbar(im, ax=ax2, label='Recall@10')
plt.tight_layout()
plt.savefig(f'{save_path}/als_analysis.png', dpi=150)
plt.show()


# Построение графиков гибридной модели
csv_path = '/content/drive/MyDrive/VK_LSVD_Results/reports/hybrid_results.csv'    # Путь к файлу
save_path = '/content/drive/MyDrive/VK_LSVD_Results/plots/'
os.makedirs(save_path, exist_ok=True)


hybrid_df = pd.read_csv(csv_path)    # Загрузка данных
print(hybrid_df.to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))    # Построение графиков


ax1 = axes[0]    # График 1: влияние alpha при разных factors
for factors in [50, 100]:
    subset = hybrid_df[hybrid_df['factors'] == factors]
    ax1.plot(subset['alpha'], subset['recall@10'],
             marker='o', label=f'factors={factors}', linewidth=2, markersize=8)
ax1.set_xlabel('Alpha (вес ALS)', fontsize=12)
ax1.set_ylabel('Recall@10', fontsize=12)
ax1.set_title('Влияние alpha на гибридную модель', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0.3, 0.5, 0.7, 0.9])


for factors in [50, 100]:    # Добавление значений на график
    subset = hybrid_df[hybrid_df['factors'] == factors]
    for _, row in subset.iterrows():
        ax1.text(row['alpha'], row['recall@10'] + 0.0001,
                f'{row["recall@10"]:.4f}', ha='center', va='bottom', fontsize=8)


ax2 = axes[1]    # График 2: тепловая карта
pivot_hybrid = hybrid_df.pivot(index='factors', columns='alpha', values='recall@10')
im = ax2.imshow(pivot_hybrid.values, cmap='Blues', aspect='auto')
ax2.set_xticks(range(len(pivot_hybrid.columns)))
ax2.set_yticks(range(len(pivot_hybrid.index)))
ax2.set_xticklabels(pivot_hybrid.columns)
ax2.set_yticklabels(pivot_hybrid.index)
ax2.set_xlabel('Alpha', fontsize=12)
ax2.set_ylabel('Factors', fontsize=12)
ax2.set_title('Тепловая карта гибридной модели', fontsize=14)


for i in range(len(pivot_hybrid.index)):    # Добавление значений в ячейки тепловой карты
    for j in range(len(pivot_hybrid.columns)):
        value = pivot_hybrid.iloc[i, j]
        ax2.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=9, color='black')


plt.colorbar(im, ax=ax2, label='Recall@10')
plt.tight_layout()
plt.savefig(f'{save_path}/hybrid_analysis.png', dpi=150)
plt.show()


# Построение графика Content-only
csv_path = '/content/drive/MyDrive/VK_LSVD_Results/reports/content_experiments.csv'    # Путь к файлу
save_path = '/content/drive/MyDrive/VK_LSVD_Results/plots/'
os.makedirs(save_path, exist_ok=True)


content_df = pd.read_csv(csv_path)    # Загрузка данных
print(content_df.to_string(index=False))


fig, ax = plt.subplots(figsize=(10, 6))    # Построение графика


ax.plot(content_df['dimension'], content_df['recall@10'],    # Линейный график
        marker='o', linewidth=2, color='royalblue', markersize=10)


for i, row in content_df.iterrows():    # Добавление значений Recall@10 на график
    ax.text(row['dimension'], row['recall@10'] - 0.00015,
            f'Recall={row["recall@10"]:.4f}', ha='center', va='top', fontsize=10, fontweight='bold')


ax.set_xlabel('Размерность эмбеддингов', fontsize=12)
ax.set_ylabel('Recall@10', fontsize=12)
ax.set_title('Влияние размерности эмбеддингов на Content-Only', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xticks([16, 32, 64])


ymax = max(content_df['recall@10']) * 1.6    # Увеличение верхней границы для аннотаций
ax.set_ylim(0, ymax)


max_recall = content_df['recall@10'].max()    # Нахождение лучших строк (максимальный Recall@10)
best_rows = content_df[content_df['recall@10'] == max_recall]


for _, row in best_rows.iterrows():    # Добавление аннотаций для всех лучших значений
    annotation_text = (f'dim={row["dimension"]}\n'
                       f'Recall={row["recall@10"]:.4f}\n'
                       f'NDCG={row["ndcg@10"]:.4f}')
   
    if row['dimension'] == 32:
        x_offset = 5
        y_offset = 0.0008
    else:  # dim == 64
        x_offset = -5
        y_offset = 0.0008
   
    ax.annotate(annotation_text,
                xy=(row['dimension'], row['recall@10']),
                xytext=(row['dimension'] + x_offset, row['recall@10'] + y_offset),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))


plt.tight_layout()
plt.savefig('/content/drive/MyDrive/VK_LSVD_Results/plots/content_analysis.png', dpi=150)
plt.show()


