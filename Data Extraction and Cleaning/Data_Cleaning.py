import pandas as pd


df = pd.read_csv(r'C:\Users\Ishant\Desktop\BE Project\Combined_99Acres_V4.csv')  # reading the csv file to be cleaned
df2 = df.copy()  # create a copy of csv file
row = 1  # initialise row counter for reading each tuple of csv

new_column = df['Location'] + str(1)
# we then add the series to the dataframe, which holds our parsed CSV file
df['Bathrooms'] = new_column

new_column = df['Bathrooms'] + str(1)
# we then add the series to the dataframe, which holds our parsed CSV file
df['Bedrooms'] = new_column

new_column = df['Bedrooms'] + str(1)
# we then add the series to the dataframe, which holds our parsed CSV file
df['Balconies'] = new_column
counter = 1
for index in range(1, 32794):
    try:
        str1 = str(df2.at[row, 'Configuration']).replace('No', str(0))  # replace no with zero
        s = map(int, filter(str.isdigit, str1))  # map the numbers from str1 and save the number in var s
        for i in s:  # storing the no of bathroom, bedroom, balconies into resp columns
            if counter == 1:
                df2.at[row, 'Bathrooms'] = i
                counter = counter + 1
                print(i)
            elif counter == 2:
                df2.at[row, 'Bedrooms'] = i
                counter = counter + 1
                print(i)
            elif counter == 3:
                df2.at[row, 'Balconies'] = i
                counter = 1
                print(i)

        print(s)
        df2.at[row, 'Floor Number'] = str(df2.at[row, 'Floor Number']).split('of', 1)[0]  # cleaning floor no column
        str3 = str(df2.at[row, 'Floor Number'])
        df2.at[row, 'Floor Number'] = map(int, filter(str.isdigit, str3))
        if str(df2.at[row, 'Parking']).find(','):
            data = str(df2.at[row, 'Parking']).split(',')
            df2.at[row, 'Parking'] = str(float(data[0])+float(data[2]))
        if str(df2.at[row, 'Parking']).find(' None '):
            df2.at[row, 'Parking'] = ''

        try:
            if str(df2.at[row, 'Price']).find('Cr'):
                k = str(df2.at[row, 'Price']).replace('Cr', '')
                df2.at[row, 'Price'] = k.replace(k, str(float(k)*100))

        except Exception as e:
            print(e)
        try:
            if str(df2.at[row, 'Price per sq.Ft']).find(','):
                df2.at[row, 'Price per sq.Ft'] = str(df2.at[row, 'Price per sq.Ft']).replace(',', '')
        except Exception as e:
            print('')
        row = row + 1
    except Exception as e:
        continue


df2.to_csv(r'C:\Users\Ishant\Desktop\Combined_99Acres_V5.csv')  # saving the data and formatting of df2 to the specified csv
print(df2.head())  # print first 5 rows and columns of df2
