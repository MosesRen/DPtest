def adult_removenull(filepath):
	csv = pd.read_csv(filepath,index_col = None,parse_dates = True,iterator = True)
	dataset = csv.read()
	a = dataset[(dataset.workclass !=' ?')]
	b = a[(a.occupation !=' ?')]
	c = b[(b.nativecountry !=' ?')]
	#分别去除workclass，occupation，nativecountry中含有空值的行
	c.to_csv(r'./dataset/adlutNoNull.csv',index=False,sep=',')
	#输出处理后的文件
adult_removenull(r'./dataset/adult.data.csv')