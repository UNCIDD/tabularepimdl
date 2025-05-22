test case for EpiRunner

1. full data with full feature columns

2. added_time = None
3. added_time = single float
4. added_time = a list of string and float
5. added_time = a list of positive float and a negative float

6. added_data = None
7. added_data =  a int
8. added_data = dataframe with a list of T not equal to added_time
9. added_data = dataframe without column T but added_time has valid t

10. initial epi data does not have column T and added_time starts with 0.0

11. Epi model to the runner is set as a string