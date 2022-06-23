def get_rows(customers, train, NUM_FILES = 10, verbose =''):
    chunk = len(customers)//NUM_FILES
    if verbose != '':
        print(f'We will split {verbose} data into {NUM_FILES} separate files.')
        print(f'There will be {chunk} customers in each file (except the last file).')
        print('Below are number of rows in each file:')
    rows = []

    for k in range(NUM_FILES):
        if k==NUM_FILES-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = train.loc[train.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows
