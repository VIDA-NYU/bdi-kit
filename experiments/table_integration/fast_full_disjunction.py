# -*- coding: utf-8 -*-
import sys
import string
import pandas as pd
import numpy as np
import glob
import csv
import os
import random
import time



def FindCurrentNullPattern(tuple1):
    current_pattern = ""
    current_nulls = 0
    for t in tuple1:
        #print(tuple1)
        if (str(t) == "nan"):
            current_pattern += "0"
            current_nulls += 1
        else:
            current_pattern += "1"
    return current_pattern, current_nulls

#used to check what are the ancestor buckets of the child bucket
def CheckAncestor(child_bucket, parent_bucket):
    for i in range(len(child_bucket)):
        if int(child_bucket[i]) == 1 and int(parent_bucket[i])==0:
            return 0
    return 1

def CheckNonNullPositions(tuple1, total_non_nulls):
    non_null_positions = set()
    for i in range(0, len(tuple1)):
        if int(tuple1[i]) == 1:
            non_null_positions.add(i)
            if len(non_null_positions) == total_non_nulls:
                return non_null_positions
    return (non_null_positions)

def GetProjectedTuple(tuple1, non_null_positions, m):
    projected_tuple = tuple()
    for j in range(0,m):
        if j in non_null_positions:
            projected_tuple += (tuple1[j],)
    return projected_tuple

#preprocess input tables
def preprocess(table):
    #table = table.replace(r'^\s*$',np.nan, regex=True)
    table.columns = map(str.lower, table.columns)
    #table = table.replace(r'^\s*$',"undefinedval", regex=True) #convert inherit nulls to "undefinedval"
    #table = table.replace(np.nan,"undefinedval", regex=True) #convert inherit nulls to "undefinedval"
    
    # table = table.applymap(str) 
    table = table.map(str) 
    table = table.apply(lambda x: x.str.lower()) #convert to lower case
    table = table.apply(lambda x: x.str.strip()) #strip leading and trailing spaces, if any
    return table


def ReplaceNulls(table, null_count):
    #table.dropna()
    #return table
    #print(table)
    null_set = set()
    for colname in table:
        for i in range (0, table.shape[0]):
            try:
                if str(table[colname][i]) == "nan":
                    table[colname][i] = "null"+ str(null_count)
                    null_set.add("null"+ str(null_count))
                    null_count += 1
            except:
                print(colname)
                #print(table[colname][i])
                print(table.shape[0])
                print(i)
                sys.exit()
    #print(null_set)
    return table, null_count, null_set

# =============================================================================
# testTablePath = r"cihr_alignment_benchmark\base tables\cihr_co-applicant_10.csv"
# testTable = pd.read_csv(testTablePath, encoding = "Latin-1")
# rtn , null_count, null_set = ReplaceNulls(testTable, 0)
# 
# =============================================================================

def AddNullsBack(table, nulls):
    columns = list(table.columns)
    input_rows = list(tuple(x) for x in table.values)
    output_rows = []
    for t in input_rows:
        new_t = tuple()
        for i in range(0, len(t)):
            if str(t[i]) in nulls:
                new_t += ("nan",)
            else:
                new_t += (t[i],)
        output_rows.append(new_t)
    final_table = pd.DataFrame(output_rows, columns =columns)
    return final_table


def CountProducedNulls(list_of_tuples):
    labeled_nulls = 0
    for row in list_of_tuples:
        for value in row:
            if value == "nan":
                labeled_nulls += 1
    return labeled_nulls


# =============================================================================
# Efficient complementation using partitioning starts here
# =============================================================================
def complementTuples(tuple1, tuple2):
    keys = 0 #find if we have common keys
    alternate1= 0 #find if we have alternate null position with non-null value in the first tuple
    alternate2 = 0 #find if we have alternate null position with non-null value in the second tuple
    newTuple = list()
    #print(tuple1)
    #print(tuple2)
    for i in range(0,len(tuple1)):
        first = str(tuple1[i])
        second = str(tuple2[i])
        if first != "nan" and second!="nan" and first != second:
            return (tuple1,False)
        elif first == "nan" and second =="nan":
            newTuple.append(first)
        elif first != "nan" and second!="nan" and first == second: #both values are equal
            keys+=1
            newTuple.append(first)
        #second has value and first is null
        elif first == "nan" and second != "nan":
            alternate1+=1
            newTuple.append(second)
        #first has value and second is null
        elif (second =="nan" and first != "nan"):
            alternate2+=1
            newTuple.append(first)
    count = 0
    for item in newTuple:
        if(item == "nan"):
            count+=1      
    if (keys >0 and alternate1 > 0 and alternate2>0 and count != len(newTuple)):
       # print(newTuple)
        return (tuple(newTuple),True)
    else:
        return (tuple(tuple1),False)


        
def PartitionTuples(table, partitioning_index):
    partitioned_tuple_dict = dict()
    all_tuples = [tuple(x) for x in table.values]
    for t in all_tuples:
        if t[partitioning_index] in partitioned_tuple_dict:
            partitioned_tuple_dict[t[partitioning_index]].append(t)
        else:
            partitioned_tuple_dict[t[partitioning_index]] = [t]
    return partitioned_tuple_dict

def GetPartitionsFromList(all_tuples, partitioning_index):
    #print(all_tuples)
    #print(partitioning_index)
    partitioned_tuple_dict = dict()
    for t in all_tuples:
        #print(t)
        if t[partitioning_index] in partitioned_tuple_dict:
            partitioned_tuple_dict[t[partitioning_index]].add(t)
        else:
            partitioned_tuple_dict[t[partitioning_index]] = {t}
    null_partition = partitioned_tuple_dict.pop(np.nan, None)
    #print(null_partition)
    if null_partition is None:
        for each in partitioned_tuple_dict:
            partitioned_tuple_dict[each] = list(partitioned_tuple_dict[each])
        return partitioned_tuple_dict
    else:
        #print("tuples in null partition:", len(null_partition))
        if len(partitioned_tuple_dict) == 0:
            partitioned_tuple_dict[np.nan] = list(null_partition)
            return partitioned_tuple_dict
        for each in partitioned_tuple_dict:
            temp_list = partitioned_tuple_dict[each]
            temp_list = temp_list.union(null_partition)
            partitioned_tuple_dict[each] = list(temp_list)            
    return partitioned_tuple_dict

def SelectPartitioningOrder(table):
    #print(table)
    statistics = dict()
    stat_unique = {}
    stat_nulls = {}
    total_rows = table.shape[0]
    #print("Total rows:", total_rows)
    unique_weight = 0
    null_weight = 1 - unique_weight #only based on null weight
    i = 0
    for col in table:
        unique_count = len(set(table[col]))
        null_count = total_rows - table[col].isna().sum()
        score = (unique_count * unique_weight) + null_count * null_weight
        statistics[i] = score
        stat_unique[i] = unique_count
        stat_nulls[i] = total_rows - null_count
        i += 1
    #print(statistics)
    stat_nulls = sorted(stat_nulls, key = stat_nulls.get, reverse = True)
    stat_unique = sorted(stat_unique, key = stat_unique.get, reverse = True)
    final_list = [stat_nulls[0]]
    stat_unique.remove(stat_nulls[0])
    final_list += stat_unique
    #return final_list    
    return sorted(statistics, key = statistics.get, reverse = True)

def FineGrainPartitionTuples(table):  
# =============================================================================
#     input_tuples = [('canada', 'diverse', "nan", "nan", "nan", "nan"),
#                   ('uk', 'temperate', "nan", "nan", "nan", "nan"),
#                   ('canada', "nan", 'toronto', 'plaza', '4', "nan"),
#                   ('canada', "nan", 'london', 'ramada', '3', "nan"),
#                   ('canada', "nan", 'london', "nan", "nan", 'air show'),
#                   ('canada', "nan", 'null0', "nan", "nan", 'mouth logan'),
#                   ('uk', "nan", 'london', "nan", "nan", 'buckingham'),
#                   ('canada', "nan", "nan", 'plaza', "nan", "nan"),
#                   ('uk', "nan", "nan", 'null1', "nan", "nan")]
# =============================================================================
    input_tuples = list({tuple(x) for x in table.values})
    partitioning_order = SelectPartitioningOrder(table)
    #partitioning_order = [1, 0, 2, 3, 4, 5]
    print("partitioning order:", partitioning_order)
    #print("total columns:", table.shape[1])
    debug_dict = {}
    list_of_list = []
    assign_tuple_id = {}
    for tid, each_tuple in enumerate(input_tuples):
        assign_tuple_id[each_tuple] = tid 
    #print(type(assign_tuple_id))            
    list_of_list.append(input_tuples)
    finalized_list = []
    #print(list_of_list)
    for i in partitioning_order:
        new_tuples = []
        track_used_tuples = {}
        print("Processing column: ", i)
        for all_tuples in list_of_list:
            if len(all_tuples) > 100:
                partitions = GetPartitionsFromList(all_tuples, i)
                #print(partitions)
                for each in partitions:
                    current_partition = partitions[each]
                    #print(current_partition)
                    create_tid = set()
                    for current_tuple in current_partition:
                        create_tid.add(assign_tuple_id[current_tuple])
                    #print(create_tid)
                    create_tid = tuple(sorted(create_tid))
                    #print(create_tid)
                    if create_tid not in track_used_tuples:
                        if len(current_partition) > 100:
                            new_tuples.append(current_partition)
                        else:
                            finalized_list.append(current_partition)
                        track_used_tuples[create_tid] = 1
            else:
                finalized_list.append(all_tuples)
        #print(new_tuples)
        print("total partitions:", len(new_tuples) + len(finalized_list))
        print("remaining partitions:", len(new_tuples))
        list_of_list = new_tuples
        debug_dict[i] = list_of_list
    if len(list_of_list) > 0:    
        finalized_list = list_of_list + finalized_list
    return finalized_list, debug_dict


def ComplementAlgorithm(tuple_list):
    receivedTuples = dict()
    for t in tuple_list:
        receivedTuples[t] = 1
    complementResults = dict()
    #itr = 1
    while (1):
        i = 1
        used_tuples = dict()
        for tuple1 in tuple_list:
            complementCount = 0
            for tuple2 in tuple_list[i:]:
                (t, flag) = complementTuples(tuple1, tuple2)
                if (flag == True):
                    complementCount += 1
                    complementResults[t] = 1
                    used_tuples[tuple2] = 1
            i += 1
            if complementCount == 0 and tuple1 not in used_tuples:
                complementResults[tuple1] = 1
        if receivedTuples.keys() == complementResults.keys():
            break
        else:
            receivedTuples = complementResults
            complementResults = dict()
            tuple_list = [tuple(x) for x in receivedTuples]
        #itr =+ 1
        #print("iteration:", itr)
    #print(complementResults)
    #print("\n")
    return [tuple(x) for x in complementResults]

def MoreEfficientComplementation(table):
    print("total tuples for complementation:", table.shape[0])
    partitioned_tuple_list, debug_dict = FineGrainPartitionTuples(table)
    #print(partitioned_tuple_list)
    complemented_list = set()
    print("Total partitions :", len(partitioned_tuple_list))
    print("Tuples in null partition:", 0)
    #print("null size:", len(null_partition))
    count = 0 
    max_partition_size = 0
    for current_partition_tuples in partitioned_tuple_list:
        current_size = len(current_partition_tuples)
        if current_size > max_partition_size:
            max_partition_size = current_size
        #print("partition number:", count + 1)
        #print("Tuples in current partition", current_size)
        complemented_tuples = ComplementAlgorithm(current_partition_tuples)
        for item in complemented_tuples:
            complemented_list.add(item)
        count +=1
        if count % 100000 == 0:
            print("partitions processed: ", count)
            print("generated tuples until now: ",len(complemented_list))
            print("Total partitions :", len(partitioned_tuple_list))
    print("largest partition size:", max_partition_size)
    return complemented_list, len(partitioned_tuple_list), max_partition_size, "full", debug_dict


# =============================================================================
# Efficient complementation using partitioning ends here
# =============================================================================


def EfficientSubsumption(tuple_list):
    #start_time = time.time_ns()
    subsumed_list = []
    m = len(tuple_list[0]) #number of columns
    bucket = dict()
    minimum_null_tuples = dict()
    bucketwise_null_count = dict()
    first_pattern, minimum_nulls = FindCurrentNullPattern(tuple_list[0])
    bucket[first_pattern] = [tuple_list[0]]
    bucketwise_null_count[minimum_nulls] = {first_pattern}
    minimum_null_tuples[minimum_nulls] = [tuple_list[0]]
    for key in tuple_list[1:]:
        current_pattern, current_nulls = FindCurrentNullPattern(key)
        if current_nulls not in bucketwise_null_count:
            bucketwise_null_count[current_nulls] = {current_pattern}
        else:
            bucketwise_null_count[current_nulls].add(current_pattern)
        if current_pattern not in bucket:
            bucket[current_pattern] = [key]
        else:
            bucket[current_pattern].append(key)
        if current_nulls < minimum_nulls:
            minimum_null_tuples[current_nulls] = [key]
            minimum_null_tuples.pop(minimum_nulls)
            minimum_nulls = current_nulls
        elif current_nulls == minimum_nulls:
            minimum_null_tuples[current_nulls].append(key)
    #output all tuples with k null values
    subsumed_list = minimum_null_tuples[minimum_nulls]
    #print(subsumed_list)
    for i in range(minimum_nulls+1, m):
        if i in bucketwise_null_count:
            related_buckets = bucketwise_null_count[i]
            parent_buckets = set()
            temp = [v for k,v in bucketwise_null_count.items()
                                    if int(k) < i]
            parent_buckets = set([item for sublist in temp for item in sublist])
            
            for each_bucket in related_buckets:
                #do something
                current_bucket_tuples = bucket[each_bucket]
                if len(current_bucket_tuples) == 0:
                    continue
                non_null_positions = CheckNonNullPositions(each_bucket, m-i)
                parent_bucket_tuples = set()
                for each_parent_bucket in parent_buckets:
                    #print(each_parent_bucket)
                    if CheckAncestor(each_bucket, each_parent_bucket) == 1:
                        list_of_parent_tuples = bucket[each_parent_bucket]
                        for every_tuple in list_of_parent_tuples:
                            projected_parent_tuple = GetProjectedTuple(
                                every_tuple, non_null_positions, m)
                            parent_bucket_tuples.add(projected_parent_tuple)
                        #print(parent_bucket_tuples)
                new_bucket_item = []     
                for each_tuple in current_bucket_tuples:
                    projected_child_tuple = set()
                    for j in range(0,m):
                        if j in non_null_positions:
                            projected_child_tuple.add(each_tuple[j])
                    projected_child_tuple = GetProjectedTuple(
                                each_tuple, non_null_positions, m)
                    if projected_child_tuple not in parent_bucket_tuples:
                        new_bucket_item.append(each_tuple)
                        subsumed_list.append(each_tuple)
                bucket[each_bucket] = new_bucket_item
    #end_time = time.time_ns()
    #print("---------------------------------")
    #print("Time taken by subsumption:",
     #     int(end_time - start_time)/10**9)
    #print("Tuples subsumed:", len(tuple_list) - len(subsumed_list))
    return subsumed_list



def FDAlgorithm(filenames, cluster):
    #stats
    print("-----x---------x--------x---")
    print("Processing cluster:", cluster)

    stats_df = pd.DataFrame(
            columns = ["cluster", "n", "s","total_cols", "f", "labeled_nulls",
                       "produced_nulls", "complement_time",
                       "complement_partitions", "largest_partition_size", "partitioning_used",
                       "subsume_time", "subsumed_tuples",
                       "total_time", "f_s_ratio"])
    m = len(filenames)
    #stats ends here
    null_count = 0
    null_set = set()
    table1 = filenames[0]
    print("Reading table:", table1)
    table1 = pd.read_csv(table1, encoding='latin1', on_bad_lines='error')
    table1 = table1.drop_duplicates().reset_index(drop=True)
    table1 = table1.replace(r'^\s*$',np.nan, regex=True)
    table1 = table1.replace("-",np.nan)
    table1 = table1.replace(r"\N",np.nan)
    if table1.isnull().sum().sum() > 0:
        #print(filenames[0])
        table1, null_count, current_null_set = ReplaceNulls(table1, null_count)
        null_set = null_set.union(current_null_set)
    table1 = preprocess(table1)
    for files in filenames[1:]:
        table2 = pd.read_csv(files, encoding='latin1', on_bad_lines='error')
        table2 = table2.drop_duplicates().reset_index(drop=True)
        table2 = table2.replace(r'^\s*$',np.nan, regex=True)
        table1 = table1.replace("-",np.nan)
        table2 = table2.replace(r"\N",np.nan)
        if table2.isnull().sum().sum() > 0:
            #print(files)
            table2, null_count, current_null_set = ReplaceNulls(table2, null_count)
            null_set = null_set.union(current_null_set)
        table2 = preprocess(table2)
        table1 = pd.concat([table1,table2])
    #print("Outer union done!")
    #measure time after preprocessing
    start_time = time.time_ns()
    #print(null_set)
    s = table1.shape[0]
    total_cols = table1.shape[1]
    #print("Total input tuples:", s)
    #print("Total input columns:", total_cols)
    schema = list(table1.columns)
    start_complement_time = time.time_ns()
    complementationResults, complement_partitions, largest_partition_size, partitioning_used, debug_dict = MoreEfficientComplementation(table1)
    end_complement_time = time.time_ns()
    complement_time = int(end_complement_time - start_complement_time)/ 10**9
    #print(schema)
    fd_table = pd.DataFrame(complementationResults, columns =schema)
    print("Adding nulls back...")
    if len(null_set) > 0:
        fd_table =  AddNullsBack(fd_table, null_set)
    print("Added nulls back...")
    #print(fd_table)
    fd_table = fd_table.replace(np.nan, "nan", regex = True)
    #print(fd_table)
    fd_data = {tuple(x) for x in fd_table.values}
    #print(fd_data)
    start_subsume_time = time.time_ns()
    subsumptionResults = EfficientSubsumption(list(fd_data))
    end_subsume_time = time.time_ns()
    subsume_time = int(end_subsume_time - start_subsume_time)/ 10**9
    subsumed_tuples = len(list(fd_data)) - len(subsumptionResults)
    fd_table = pd.DataFrame(subsumptionResults, columns =schema)
    fd_table = fd_table.replace(np.nan, "nan", regex = True)
    fd_data = [tuple(x) for x in fd_table.values]
    print("Output tuples: ( total", len(fd_data),")")
    for t in fd_data:
        print(t)
    end_time = time.time_ns()
    f = len(fd_data)
    produced_nulls = CountProducedNulls(fd_data)
    total_time = int(end_time - start_time)/10**9
    #print("---------------------------------")
    #print("Time taken FD algorithm:", total_time)
    #print("Tuples generated FD algorithm:", f)
    append_list = [cluster, m, s, total_cols, f, len(null_set),
                   produced_nulls, complement_time, 
                   complement_partitions, largest_partition_size,
                   partitioning_used, subsume_time,
                   subsumed_tuples, total_time, f/s]
    a_series = pd.Series(append_list, index = stats_df.columns)
    # print(a_series)
    # stats_df = stats_df.append(a_series, ignore_index=True)    
    stats_df = pd.concat([stats_df, a_series], ignore_index=True)  
    return fd_table, stats_df, debug_dict

statistics = pd.DataFrame(
            columns = ["cluster", "n", "s", "f", "labeled_nulls",
                       "produced_nulls", "complement_time",
                       "complement_partitions", "largest_partition_size", "partitioning_used",
                       "subsume_time",
                       "subsumed_tuples", "total_time", "f_s_ratio"])


