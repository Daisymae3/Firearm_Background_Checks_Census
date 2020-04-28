#!/usr/bin/env python
# coding: utf-8

# # Project: Investigating Firearm Background Checks with U.S. Census Data
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# In this analysis, I will be investigating firearm background checks against data from the United States Census. The data was obtained from Udacity's Data Analyst Nanodegree website. Further information for the firearm background checks can be found on the FBI National Instant Criminal Background Check (NICS) website at https://www.fbi.gov/services/cjis/nics. The data was originally in PDF form but has been converted to CSV file via the GitHub repository https://github.com/BuzzFeedNews/nics-firearm-background-checks/blob/master/README.md. The U.S. Census contains data from 2016 and can be found on the Census.gov website https://www.census.gov/. 
# 
# According to the FBI NICS website:  
# 
#     "Mandated by the Brady Handgun Violence Prevention Act (Brady Act) of 1993, Public Law 103-159, the National Instant Criminal Background Check System (NICS) was established for Federal Firearms Licensees (FFLs) to contact by telephone, or other electronic means, for information to be supplied immediately on whether the transfer of a firearm would be in violation of Section 922 (g) or (n) of Title 18, United States Code, or state law. The Brady Act is a public record and is available from many sources, including the Internet at www.atf.gov."
# 
# The process of the NICS firearm background checks is as follows:
# 
#     "When a person tries to buy a firearm, the seller, known as a Federal Firearms Licensee (FFL), contacts NICS electronically or by phone. The prospective buyer fills out the ATF form, and the FFL relays that information to the NICS. The NICS staff performs a background check on the buyer. That background check verifies the buyer does not have a criminal record or isn't otherwise ineligible to purchase or own a firearm"
# 
# The purpose of this analysis is to explore:
# 
# How many firearm background checks are performed by each ethnicity within the U.S.?
# 
# How many firearm background checks are performed by different education levels in the U.S.?
# 
# Is there a correlation between gun background checks compared to percent of people in poverty?
# 
# Which states have the highest growth of gun background checks within the data time period?
# 
# What is the overall trend of gun background checks thoughout the data time period?
# 
# It is important to note that the statistics within this chart represent the number of firearm background checks initiated through the NICS. They do not represent the number of firearms sold. Based on varying state laws and purchase scenarios, a one-to-one correlation cannot be made between a firearm background check and a firearm sale.

# In[131]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[132]:


gun = pd.read_csv('gun_data.csv')
census = pd.read_csv('census_data.csv')
gun.head()


# The transactions types in the firearm dataset are unnecessary for the analyses that I'm performing. These will need to be dropped.

# In[133]:


gun.shape


# I will only need three of these columns for my analyses.

# In[134]:


gun.info()


# There is missing data in the transaction columns, but these columns will be dropped before performing my analyses, so there is no need to change them.

# In[135]:


print(gun.duplicated().sum())


# There is no duplicate data for the Gun dataset.

# #### Now we'll explore the Census Data

# In[136]:


census.head(20)


# To make this dataset easier to analyse, I will transpose the rows and columns. The column headers will need to be formatted. Some of the proportions have a non numerical value of 'Z', which will need to converted to numerical format. Also, the proportions are in different formats (decimal and percentage) and will need to be converted to a uniform numerical format.

# In[137]:


census.info()


# It does not look like there are any null values for any of the states in the Census data. 

# In[138]:


print(census.duplicated().sum())


# There are three duplicate rows in the Census dataset.

# In[139]:


dups = census[census.duplicated()]
dups


# Duplicate rows of NaN values need to be dropped.

# In[140]:


census.drop_duplicates(inplace=True)


# In[141]:


print(census.duplicated().sum())


# All duplicate data have been dropped from the Census dataframe.

# ### Data Cleaning: Formatting column names, changing datatypes, and dropping unnecessary data.

# In[142]:


gun.drop(gun.iloc[:, 2:26], inplace = True, axis = 1)
gun.head()


# My analyses do not require the specific transaction type, so the transaction columns have been dropped.

# In[143]:


gun['year'] = pd.DatetimeIndex(gun['month']).year
gun['month_no'] = pd.DatetimeIndex(gun['month']).month
gun.head()


# I split the month column into separate month_no and year columns to make analysing easier.

# In[144]:


cols = list(gun.columns.values) 
cols.pop(cols.index('year'))
cols.pop(cols.index('state'))
cols.pop(cols.index('totals')) 
gun = gun[cols+['year','state','totals']] 
gun.head()


# The columns have been reordered.

# In[145]:


gun = gun.drop(['month'], axis=1)
gun.head()


# The old month column has been dropped from the Gun dataset.

# In[146]:


gun.info()


# The datatypes for the columns are correct.

# In[147]:


gun_states = gun.groupby('state').totals.mean()
gun_states.shape


# The number of states in the Gun Dataset do not match the Census Dataset.

# In[148]:


gun = gun.drop(gun[gun.state.isin(['Guam', 'District of Columbia', 'Mariana Islands', 'Puerto Rico', 'Virgin Islands'])].index)
gun_states = gun.groupby('state').totals.mean()
gun_states.shape


# The datasets now have states that match

# #### Now we'll clean up the Census Data

# In[149]:


census = census.T
census.head(50)


# The dataset was transposed to make it easier for cleaning and analysis.

# In[150]:


census.columns = census.iloc[0]
census.head()


# The index numbers were replaced by the first row as the column headers.

# In[151]:


census= census.drop(census.index[0:2])
census.head()


# The first two rows were unnecessary and dropped.

# In[152]:


census.reset_index(drop = False, inplace = True)
census.head()


# Index numbers were reset for the rows.

# In[153]:


census = census.rename_axis(None, axis=1)
census.head()


# The index row name 'Fact' was removed.

# In[154]:


census = census.rename(columns = {'index':'State'})
census.head()


# The column labeled as index was renamed 'State'.

# In[155]:


for i, v in enumerate(census.columns):
    print(i,v)


# Checking to see which column index numbers I need for my analyses.

# In[156]:


census = census.iloc[:, np.r_[0,1, 13:21, 35:37, 50]]
census.head()


# Unnecessary columns were filtered out.

# In[157]:


census.columns = census.columns.str.split(',').str[0]
census.head()


# Unnecessary characters were removed and the column names were shortened.

# In[158]:


census.columns.values[[2, 9]] = ['White alone', 'White alone non hispanic']
census.head()


# After the column names were split, it produced two columns with the same name, so I renamed "White alone non Hispanic", insead of "White alone". These will be combined and averaged at the end of the data cleaning.

# In[159]:


census.columns = census.columns.str.strip().str.lower().str.replace(' ', '_')
census.head()


# The column names were formatted into lower case with no spaces.

# In[160]:


census.apply(lambda column: column.astype(str).str.contains('Z').any(), axis=0)


# The column that contains the string 'Z' has been identified.

# In[161]:


census['native_hawaiian_and_other_pacific_islander_alone']


# The native hawaiian column contains 4 instances of the string 'Z'.

# In[162]:


census = census.replace(to_replace = "Z", value = '0.0%') 
census['native_hawaiian_and_other_pacific_islander_alone']


# According to the Census Data, the value 'Z' means that the value is greater than zero but less than half unit of measure shown. I converted 'Z' to a 0.0% (str) and will then convert all strings to numeric formats.

# In[163]:


census.apply(lambda column: column.astype(str).str.contains('Z').any(), axis=0)


# All strings of 'Z' have been replaced.

# In[164]:


census.info()


# The columns of the ethnicity, type of degree, and poverty levels need to be converted from strings to floats.

# In[165]:


census.applymap(lambda x: '%' in str(x))


# The '%' is used in some of the columns, but not all.

# In[166]:


for i, v in enumerate(census.columns):
    print(i,v)


# Index numbers have been identified with the '%' sign.

# In[167]:


census.iloc[np.r_[0:30, 42:50], 2:] = census.iloc[np.r_[0:30, 42:50], 2:].replace('%','',regex=True).astype('float')/100
census


# '%' string was filtered and percentage was converted to decimal form.

# In[168]:


census.applymap(lambda x: '%' in str(x))


# All '%' signs seem to have been converted.

# In[169]:


census.population_estimates = census.population_estimates.replace(",",'', regex=True).astype(int)
census


# The column 'population_estimate' has been converted to integer.

# In[170]:


census.dtypes


# The data types of the ethncities, type of degree, and poverty are still strings and need to be changed to a numeric form.

# In[171]:


cols = census.columns.drop('state', 'population_estimates')
census[cols] = census[cols].apply(pd.to_numeric, errors='coerce')
census.info()


# The above mentioned strings have been converted to floats. I used a to_numeric function to avoid scientific notations in my data set.

# In[172]:


census['white_alone_mean'] = census['white_alone']+census['white_alone_non_hispanic']
census['white_alone_mean'] = census['white_alone_mean']/2
census = census.drop(['white_alone', 'white_alone_non_hispanic'], axis=1)
census.head()


# The 'white_only' and 'white_only_non_hispanic' columns have been combined and averaged in a new column. The old columns have been dropped.

# In[173]:


cols = list(census.columns.values) 
cols.pop(cols.index('high_school_graduate_or_higher')) 
cols.pop(cols.index("bachelor's_degree_or_higher")) 
cols.pop(cols.index('persons_in_poverty'))
census = census[cols+['high_school_graduate_or_higher',"bachelor's_degree_or_higher", 'persons_in_poverty']] 
census.head()


# The Census Data Columns have been reordered.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1: How many firearm background checks are performed by each ethnicity within the U.S.?

# In[174]:


bg_checks_mean = int(gun['totals'].mean())
bg_checks_mean


# A variable has been created for the average total of Firearm Background Checks performed per state per month.

# In[175]:


african = census['black_or_african_american_alone'].mean()*bg_checks_mean
asian = census['asian_alone'].mean()*bg_checks_mean
hawaiian = census['native_hawaiian_and_other_pacific_islander_alone'].mean()*bg_checks_mean
hispanic = census['hispanic_or_latino'].mean()*bg_checks_mean
native_a = census['american_indian_and_alaska_native_alone'].mean()*bg_checks_mean
two_plus = census['two_or_more_races'].mean()*bg_checks_mean
white = census['white_alone_mean'].mean()*bg_checks_mean


# The variables were created by multiplying the mean proportion of each ethnicity for all states by the mean number of background checks performed per state per month in the U.S.

# In[176]:


plt.subplots(figsize=(8, 5))
locations = [1, 2, 3, 4, 5, 6, 7]
heights = [african, asian, hawaiian, hispanic, native_a, two_plus, white]
labels = ['African', 'Asian', 'Hawaiian', 'Hispanic', 'Native Indian/Alaskan', 'Multi Racial', 'Caucasion']
plt.bar(locations, heights, tick_label=labels)
plt.xticks(rotation=90)
plt.title('Average Monthly Firearm Background Checks per Ethnicity per U.S. State')
plt.xlabel('Ethnicity')
plt.ylabel('Background Checks');


# Bar chart shows the mean number of firearm background checks performed per ethnicity per state per month in the U.S.

# ### Research Question 2:  How many firearm background checks are performed per education level within the U.S.?

# In[177]:


diploma = census['high_school_graduate_or_higher'].mean()*bg_checks_mean
bachelors = census["bachelor's_degree_or_higher"].mean()*bg_checks_mean


# The variables were created by multiplying the mean proportion of each degree for all states by the mean number of background checks performed per state per month in the U.S.

# In[178]:


locations = [1, 2]
heights = [diploma, bachelors]
labels = ['High School Diploma', "Bachelor's Degree"]
plt.bar(locations, heights, tick_label=labels)
#plt.xticks(rotation=90)
plt.title('Average Monthly Firearm Background Checks per Degree per U.S State')
plt.xlabel('Education Level')
plt.ylabel('Background Checks');


# Bar chart shows the mean number of firearm background checks performed per education level per month in the U.S.

# ### Research Question 3: Is there a correlation between gun background checks compared to the percentage of people in poverty?

# In[179]:


state_checks = gun.loc[:, ['state', 'totals']].groupby('state').mean()
state_checks.head()


# A groupby function was performed to get the necessary data for the next analysis. These give the average number of permits sold in each individual state per month in the U.S.

# In[180]:


merged = pd.merge(state_checks, census, on='state', how='inner')
merged.head()


# An inner merge was performed on the 'state' column to make analysis easier.

# In[181]:


state_checks_mean = census['persons_in_poverty']*merged['totals']
percent_poverty = census['persons_in_poverty']*100


# Two variables were created. 
# 
# The first was by multiplying the proportion of persons in poverty in each state by the average number of background checks performed in the same state. 
# 
# The second variable was created by multiplying the proportion of people in poverty by 100 to get a percentage of
# poverty in each individual state.

# In[182]:


plt.subplots(figsize=(8, 5))
plt.scatter(percent_poverty, state_checks_mean)
plt.title('Average Monthly Firearm Background Checks per Proportion of Poverty in each U.S. State')
plt.xlabel('Percent Povert')
plt.ylabel('Background Checks');


# In[183]:


merged.sort_values(by=['totals', 'persons_in_poverty'], ascending=False).head()


# It looks like Kentucky is the major outlier in this chart.

# ### Research Question 4: Which states have the highest growth of firearm background checks within the data time period?

# In[184]:


years_sum = gun.groupby(['state', 'year'])['totals'].sum().reset_index()
years_sum.head()


# The monthly background checks were summed to get the total for each year.

# In[185]:


checks_99 = years_sum.query('year == 1999')
checks_16 = years_sum.query('year == 2016')


# The years 1999 and 2016 were filtered used because they had complete monthly data.

# In[186]:


diff = (checks_16.set_index('state')-checks_99.set_index('state')).dropna(axis=0)
diff.head()


# The difference was achieved by subtracting the 1999 data from the 2016 data.

# In[187]:


diff = diff.reset_index(drop = False)


# The index numbers were reset after previous queries.

# In[188]:


top_five = diff.sort_values(by=['totals'], ascending=False).head()
top_five


# States were sorted in descending order based on totals.

# In[189]:


top_five = top_five.iloc[0:5, [0, 2]]
top_five


# The top five states were filtered out.

# In[190]:


Kentucky = top_five.iloc[0, 1]
California = top_five.iloc[1, 1]
Illinois = top_five.iloc[2, 1]
Indiana = top_five.iloc[3, 1]
Florida = top_five.iloc[4, 1]


# Variables were created for each of the top 5 states.

# In[191]:


locations = [1, 2, 3, 4, 5]
heights = [Kentucky, California, Illinois, Indiana, Florida]
labels = ['Kentucky', 'California', 'Illinois', 'Indiana', 'Florida']
plt.bar(locations, heights, tick_label=labels)
plt.title('States with the Highest Growth of Yearly Firearm Background Checks Between 1999 and 2016')
plt.xlabel('State')
plt.ylabel('Background Checks');


# Bar chart shows the states with the highest growth of firearm background checks between 1999 and 2016.

# ### Research Question 5: What is the overall trend of firearm background checks thoughout the data time period.

# In[192]:


time_years = years_sum.iloc[1:19, :].groupby('year')['totals'].sum().reset_index()
time_years.head()


# Filtered the years to include only years that had complete monthly data.

# In[193]:


year = time_years['year']
year_total = time_years['totals']


# Variables were created for the year and the total number of background checks for that year.

# In[194]:


plt.subplots(figsize=(8, 5))
plt.plot(year, year_total)
plt.title('Average Number of Firearm Background Checks Per Year Between 1999-2016 in the U.S')
plt.xlabel('Year')
plt.ylabel('Background Checks')
plt.locator_params(nbins=11)


# A chart of average number of background checks per year.

# <a id='conclusions'></a>
# ## Conclusions
# 
# How many firearm background checks are performed by each ethnicity within the U.S.?
# 
# The chart shows that caucasion americans produce the most firearm background checks of all ethnicities in the United States at approximately 17,500 checks per month per state. However, this implies that all ethnicities undergo firearm background checks in the same proportions that they are present in the United States. This assumption may lead to innacurate results in the analysis. Some ethnicities may be more prone to private gun sales over public, or vice versa.
# 
# How many firearm background checks are performed by different education levels in the U.S.?
# 
# The chart shows that people with high school diplomas produce more firearm background checks than those holding a Bachelor's Degree, with approximately 21,000 checks per month per state. Again, this implies that the different education levels undergo gun background checks in the same proportions that they are present in the United States. This is a limitation in the analysis.  
# 
# Is there a correlation between firearm background checks compared to the percentage of people in poverty?
# 
# According the scatter plot, there is a positive correlation to the percentage of poverty to the number of monthly firearm background checks performed per state. The states with the higher rates of poverty tend to have the higher rates of firearm background checks performed. A major outlier in this plot is Kentucky. A possible explanation for this may be found from an article printed by WFPL in 2015 that states:
# 
#     "one reason Kentucky reports such a high number of checks annually — more than double the number of Texas and California combined so far this year — is a policy that requires automatic monthly background checks on every holder of concealed-carry permits in the commonwealth." The article can be found on the wfpl website https://wfpl.org/kentucky-background-checks-stand-out/
# 
# Which states have the highest growth of firearm background checks within the data time period?
# 
# The graph shows that Kentucky has the highest growth of firearm background checks for all the states in the U.S. at approximately 3.5 million firearms background checks since 1999. The previously mentioned article which states that the state of Kentucky rechecks concealed carry permit holders on a monthly basis may be a reason why Kentucky has more than twice the firearm checks compared to California, at approximately 1.5 million.
# 
# What is the overall trend of gun background checks thoughout the data time period?
# 
# The number of firearm background checks remained relatively unchanged between 1999 and 2007. The year 2007 saw a jump in background checks, but decreased slightly in 2010. There was a signigicant spike in background checks between 2010 and 2015, which increased from approximately 300000 per year to over 700000 per year. More research is needed to speculate the reason behind the increased firearm background checks within those years.

# In[195]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

