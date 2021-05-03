import copy
import csv
import json
from tabulate import tabulate

import sys

csv.field_size_limit(sys.maxsize)


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Try to sel col_index to the index of the col_dentifier
        try:
            col_index = self.column_names.index(col_identifier)
        # If invalid ID, print to console
        except ValueError:
            print(col_identifier, "is not a valid column id!")

        # Set length of data to iterate
        n = len(self.data)

        # Initialize empty array to append NA vals
        list = []

        # Iterate through self.data for n(data length) times at index i
        for i in range(n):

            if (self.data[i][col_index] == "NA"):
                if (include_missing_values):
                    list.append("NA")
            else:
                # append self.data at [i][col] to list
                list.append(self.data[i][col_index])

        return list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        row, col = self.get_shape()
        for i in range(row):
            for j in range(col):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except:
                    self.data[i][j] = self.data[i][j]

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """

        for r in rows_to_drop:
            for row in self.data:
                if row == r:
                    self.data.remove(row)
                    break

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            count = 0
            output_count = 2
            for row in reader:
                output_count += 1
                if count == 0:
                    self.column_names = row
                else:
                    self.data.append(row)
                count += 1
        self.convert_to_numeric()
        return self

    def load_from_json_file(self, filename, rows=None):
        """Loads data from json file. 

        Args:
            filename(str): name of the json file to read
            rows(int): if None, function will return all rows. Otherwise, returns number of rows specified as integer

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        """
        with open(filename, encoding='utf8') as f:
            if rows:
                index = 0
                for row in f:
                    row = json.loads(row)
                    if len(self.column_names) < 1:
                        self.column_names = list(row.keys())
                    self.data.append(list(row.values()))
                    if index >= rows:
                        break
                    index += 1
            else:
                for row in f:
                    row = json.loads(row)
                    if len(self.column_names) < 1:
                        self.column_names = list(row.keys())
                    self.data.append(list(row.values()))
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, mode="w", newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        rows, cols = self.get_shape()
        indices = []
        for n in key_column_names:
            for i in range(cols):
                if self.column_names[i] == n:
                    indices.append(i)
        seen = []
        equal = False

        for cols in key_column_names:
            for i in range(rows):
                for ii in range(i+1, rows):
                    equal = self.list_equal(
                        self.data[i], self.data[ii], indices)
                    if equal:
                        if self.data[ii] not in duplicates:
                            duplicates.append(self.data[ii])
                            break
        return duplicates

    def list_equal(self, d1, d2, indices):
        '''
            Function to check if the given lists are equal based on given indices

            Args:
                d1 First list
                d2 second list
                indices list of indexes of columns

            Returns: 
                False if lists are not equal, true otherwise
        '''

        for i in indices:
            if d1[i] != d2[i]:
                return False
        return True

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        exist = True

        while exist:
            exist = False
            for row in self.data:
                if "NA" in row:
                    self.data.remove(row)
                    exist = True

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        idx = -1
        for i in range(len(self.column_names)):
            if col_name == self.column_names[i]:
                idx = i
        column = self.get_column(col_name)
        sum = 0
        count = 0
        for i in column:
            if isinstance(i, float):
                sum = sum + i
                count += 1
        if count > 0:
            avg = sum / count
        else:
            avg = 0

        for row in self.data:
            for i in range(len(row)):
                if row[i] == "NA":
                    row[i] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """

        table = MyPyTable()
        table.column_names = col_names
        self.convert_to_numeric()
        for s in col_names:
            self.replace_missing_values_with_column_average(s)

        d = []
        rows, cols = self.get_shape()

        if rows > 0:
            for i in range(len(col_names)):
                try:
                    col = self.get_column(col_names[i])
                    min_val = min(col)
                    max_val = max(col)
                    total = sum(col)
                    count = len(col)
                    mid = (min_val + max_val) / 2.0
                    col.sort()
                    med = 0
                    m = (count-1) // 2

                    if count % 2 == 0:
                        med = (col[m] + col[m + 1]) / 2
                    else:
                        med = col[m]

                    d.append([col_names[i], min_val, max_val,
                              mid, total / count, med])
                except:
                    x = 0  # dummy statment
        table.data = d
        return table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Create an empty array to store column names
        names = []
        for s in self.column_names:
            names.append(s)
        for s in other_table.column_names:
            if s not in key_column_names:
                names.append(s)

        indices1 = []
        indices2 = []

        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[j]:
                    indices1.append(j)

            for j in range(len(other_table.column_names)):
                if key_column_names[i] == other_table.column_names[j]:
                    indices2.append(j)

        d = []
        count = 0
        for row1 in self.data:
            for row2 in other_table.data:
                if self.is_equal(row1, row2, indices1, indices2):
                    d.append(row1)
                    for i in range(len(row2)):
                        if i not in indices2:
                            d[count].append(row2[i])
                    count += 1
        table = MyPyTable()
        table.column_names = names
        table.data = d

        return table

    def is_equal(self, r1, r2, indices1, indices2):
        """Return bool if match is found

        Args:
            self, row1, row2, indices1, indices2 to iterate through both and check for equivalence

        Returns:
            Bool (True) if equivalence found (False) if not
        """
        # Iterate through to check value equivalence
        for i in range(len(indices1)):
            if r1[indices1[i]] != r2[indices2[i]]:
                return False
        return True

    def has_all(self, row1, row2, index):
        """Returns a bool based on value found or not

        Args:
            self, row1, row2, and an index

        Returns:
            Bool (True) if value is found in row (False) if not
        """
        for i in index:
            if row2[i] not in row1:
                return False
        return True

    def get_index(self, row, col):
        """
        Get the indes of a particular element

        Args:
            Self, Row, Col

        Returns:
            index location
        """
        for i in range(len(row)):
            if row[i] == col:
                return i
        return -1

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        '''
        names = []
        for s in self.column_names:
            names.append(s)
        for s in other_table.column_names:
            if s not in key_column_names:
                names.append(s)

        indices1 = []
        indices2 = []

        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[j]:
                    indices1.append(j)

            for j in range(len(other_table.column_names)):
                if key_column_names[i] == other_table.column_names[j]:
                    indices2.append(j)

        d = []
        count = 0        
        for row1 in self.data:
            for row2 in other_table.data:
                if self.is_equal(row1, row2, indices1, indices2):
                    d.append(row1)
                    for i in range(len(row2)):
                        if i not in indices2:
                            d[count].append(row2[i])
                    count += 1
        
        '''

        new_column_names = []
        joined_data = []

        # Create deep copy arrays
        other_atts = copy.deepcopy(other_table.column_names)
        self_atts = copy.deepcopy(self.column_names)

        # Make deep copy
        self_data = copy.deepcopy(self.data)

        # Iterate through to insert non NA values
        for name in key_column_names:
            # Remove the keys
            other_atts.remove(name)
            self_atts.remove(name)

        new_column_names = self.column_names + other_atts  # Make column list

        index_array = []

        # Iterate through table
        for self_row in self_data:
            self_keys = []  # Initialize list to get key values in self row
            found_match = False

            # Iterate over names to get kvs
            for name in key_column_names:
                index = self.column_names.index(name)  # Get index
                val = self_row[index]  # Get value
                self_keys.append(val)  # Append value

            match_index = 0
            # Iterate through other tables
            for other_row in other_table.data:
                other_keys = []  # Initialize list to get key values in other row

                # Iterate through names to get KVs
                for name in key_column_names:
                    index = other_table.column_names.index(name)  # Get index
                    val = other_row[index]  # Get value
                    other_keys.append(val)  # Append Value

                # Check for KV match
                if self_keys == other_keys:
                    index_array.append(match_index)
                    found_match = True

                    # Successful Match
                    other_copy = copy.deepcopy(other_row)

                    # Remove Duplicate
                    for val in other_keys:
                        other_copy.remove(val)

                    # Append rows
                    row_tba = self_row + other_copy

                    # Append to table
                    joined_data.append(row_tba)

                match_index = match_index + 1

            if (not found_match):
                # Add self row to outter
                row_tba = self_row

                # Add NA values
                for _ in other_atts:
                    row_tba.append("NA")

                # Append rows
                joined_data.append(row_tba)

        na_indices = []  # List to hold indices that will have NAs in other table
        keep_indices = []  # List to hold the indices we want to keep in other table

        # Build up na and keep indices lists
        for i in range(len(new_column_names)):
            if not new_column_names[i] in other_table.column_names:
                # The column name is not in the other table, will fill with NA
                na_indices.append(i)
            else:
                # The column name is in the other table, want to keep value
                keep_indices.append(
                    other_table.column_names.index(new_column_names[i]))

        # Deep copy to alter data
        other_data = copy.deepcopy(other_table.data)

        # Delete unecessary data
        for row in other_data:
            for i in range(len(row)):
                if not i in keep_indices:
                    del row[i]

        # Add NAs where needed
        for row in other_data:
            for i in na_indices:
                row.insert(i, "NA")

        # Select correct rows
        for i in range(len(other_data)):
            if not i in index_array:
                joined_data.append(other_data[i])

        return MyPyTable(new_column_names, joined_data)

    def head(self, range_=5):
        """Pretty prints range of data selected from the start of the table."""
        print(tabulate(self.data[:range_], headers=self.column_names))

    def tail(self, range_=5):
        """Pretty prints range of data selected from the end of the table."""
        print(tabulate(self.data[-range_:], headers=self.column_names))
