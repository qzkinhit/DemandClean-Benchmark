import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, monotonically_increasing_id
from AnalyticsCache.getScore import calculate_accuracy_and_recall_spark
from Clean import CleanonCloud
from SampleScrubber.cleaner.multiple import AttrRelation

cleaners = [
    AttrRelation(['establishment_date'], ['establishment_time'], '1'),
    AttrRelation(['registered_capital'], ['registered_capital_scale'], '2'),
    AttrRelation(['enterprise_name'], ['industry_third'], '3'),
    AttrRelation(['enterprise_name'], ['industry_second'], '4'),
    AttrRelation(['enterprise_name'], ['industry_first'], '5'),
    AttrRelation(['industry_first'], ['industry_second'], '6'),
    AttrRelation(['industry_second'], ['industry_third'], '7'),
    AttrRelation(['annual_turnover'], ['annual_turnover_interval'], '8'),
    AttrRelation(['latitude', 'longitude'], ['province'], '9'),
    AttrRelation(['latitude', 'longitude'], ['city'], '10'),
    AttrRelation(['latitude', 'longitude'], ['district'], '11'),
    # AttrRelation(['latitude', 'longitude'], ['enterprise_address'], '22'),
    AttrRelation(['enterprise_address'], ['province'], '12'),
    AttrRelation(['enterprise_address'], ['city'], '13'),
    AttrRelation(['enterprise_address'], ['district'], '14'),
    AttrRelation(['enterprise_address'], ['latitude'], '15'),
    AttrRelation(['enterprise_address'], ['longitude'], '16'),
    AttrRelation(['province'], ['city'], '17'),
    AttrRelation(['city'], ['district'], '18'),
    AttrRelation(['enterprise_name'], ['enterprise_type'], '19'),
    AttrRelation(['enterprise_id'], ['enterprise_name'], '20'),
    AttrRelation(['social_credit_code'], ['enterprise_name'], '21')
]

table_name = 'ai4data_enterprise_bak'
clean_table_name = 'ai4data_enterprise_bak_anomaly_data_flag'
dirty_table_name = 'ai4data_enterprise_bak_preH'
data_name=table_name + '_100w'
save_table_name = data_name+ '_cleaned'
database_name = 'tid_sdi_ai4data'  # 设置数据库名称变量
# 指定比对的属性集合
attributes = attributes_set = [
    'annual_turnover',
    'annual_turnover_interval',
    'city',
    'district',
    'enterprise_address',
    'enterprise_id',
    'enterprise_name',
    'enterprise_type',
    'establishment_date',
    'establishment_time',
    'industry_first',
    'industry_second',
    'industry_third',
    'latitude',
    'longitude',
    'province',
    'registered_capital',
    'registered_capital_scale',
    'social_credit_code'
]
# 自定义索引列名
index_name = 'enterprise_id'  # 替换为实际索引列名
single_max=20000

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("DataCleaning") \
        .config("spark.sql.session.state.builder", "org.apache.spark.sql.hive.UQueryHiveACLSessionStateBuilder") \
        .config("spark.sql.catalog.class", "org.apache.spark.sql.hive.UQueryHiveACLExternalCatalog") \
        .config("spark.sql.extensions", "org.apache.spark.sql.DliSparkExtension") \
        .config("spark.sql.hive.implementation", "org.apache.spark.sql.hive.client.DliHiveClientImpl") \
        .enableHiveSupport() \
        .getOrCreate()
    query = f"select * from {database_name}.{table_name} ORDER BY enterprise_address DESC LIMIT 1000000"
    # query = f"SELECT * FROM {database_name}.{table_name} LIMIT 1000000"
    data = spark.sql(query)
    # 添加数据行的索引,检查是否存在 'index' 列
    if 'index' not in data.columns:
        # 添加 'index' 列
        data = data.withColumn("index", monotonically_increasing_id())
    # 持久化 DataFrame
    data.persist()
    elapsed_time = 0
    start_time = time.perf_counter()
    data = CleanonCloud(spark, cleaners, data, data_name, database_name,single_max=single_max)
    end_time = time.perf_counter()
    elapsed_time += end_time - start_time
    print(f"当前清洗总执行时间: {elapsed_time:.4f} 秒")

    print("完成清洗，保存清洗结果，写入数据大小: " + str(data.count()))
    data = data.withColumn("cleaned", lit(True))

    data.write.mode('overwrite').saveAsTable(f"{database_name}.{save_table_name}")

    print("验证清洗性能:")
    clean_data_query = f"SELECT * FROM {database_name}.{clean_table_name}"
    dirty_data_query = f"SELECT * FROM {database_name}.{dirty_table_name}"
    clean_data = spark.sql(clean_data_query)
    dirty_data = spark.sql(dirty_data_query)
    cleaned_data = data

    accuracy, recall = calculate_accuracy_and_recall_spark(clean_data, dirty_data, cleaned_data, attributes, index_name)

    print(f"修复准确率: {accuracy}")
    print(f"修复召回率: {recall}")

    spark.stop()
