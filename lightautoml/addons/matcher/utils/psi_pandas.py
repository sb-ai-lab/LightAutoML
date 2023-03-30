import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('psi_pandas')
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format='[%(asctime)s | %(name)s | %(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.INFO
)


class psi:
    """Calculates population stability index for buckets

    For numeric data - numeric buckets (except when numeric column
    includes only NULL). For categorical data:
    for n<20 bucket equals proportion of each category,
    for n>20 bucket equals to a group of categories,
    for n>100 it calculates unique_index based on Jaccard similarity,
    but in case of disbalance null-good data returns PSI

    Args:
        expected - expected values: spark dataframe
        actual - actual values: spark dataframe
        column_name: str
        plot - distribution plot if True. Default=False

    Returns:
        psi_value - PSI for column: float
        psi_dict - input in PSI for each bucket: dict
        new_cats - new categories
        (for not categorical data inapplicable - returns empty list): list
        abs_cats - categories that absents in actual column
        (for not categorical data inapplicable - returns empty list): list

    """
    
    def __init__(self, expected, actual, column_name, axis=1, plot=False):
        self.expected = expected[column_name].values
        self.actual = actual[column_name].values
        self.actual_len = len(self.actual)
        self.expected_len = len(self.expected)
        self.column_name = column_name
        self.column_type = self.expected.dtype
        self.expected_shape = self.expected.shape
        self.expected_nulls =  np.sum(pd.isna(self.expected))
        self.actual_nulls = np.sum(pd.isna(self.actual))
        self.axis = 1
        self.plot = plot
        if self.column_type in [np.dtype('O')]:
            self.expected_uniqs = expected[column_name].unique()
            self.actual_uniqs = actual[column_name].unique()
    
    def jac(self):
        """Calculates Jacquard similarity

        Jacquard similarity measures the intersection between two sequences
        versus the union of tho sequences

        Returns:
            Jacquard similarity: float

        """
        x = set(self.expected_uniqs)
        y = set(self.expected_uniqs)
        logger.info(f'Jacquard similarity is {round(len(x.intersection(y)) / len(x.union(y)), 6)}')
        return len(x.intersection(y)) / len(x.union(y))
    
    
    def plots(self, nulls, expected_percents, actual_percents, breakpoints, intervals):
        """Plots expected and actual percents

        Args:
            nulls: ???
            expected_percents: float
            actual_percents: float
            breakpoints: {__iter__}
            intervals: {__len__}

        """
        points = [i for i in breakpoints] 
        plt.figure(figsize=(15,7))
        plt.bar(np.arange(len(intervals))-0.15, expected_percents, label='expected', alpha=0.7, width=.3)
        plt.bar(np.arange(len(intervals))+0.15, actual_percents, label='actual', alpha=0.7, width=.3)
        plt.legend(loc='best')
        if self.column_type not in [np.dtype('O')]:
            plt.xticks(range(len(intervals)), intervals, rotation=90)
        else: 
            plt.xticks(range(len(points)), points, rotation=90)
        plt.title(self.column_name)
        #plt.savefig(f"C:\\Users\\Glazova2-YA\\Documents\\data\\bip\\summary_psi_plots\\{self.column_name}.png")
        plt.show()
        
        
        
    def sub_psi(self, e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        logger.info(f'sub_psi value is {round(value, 6)}')
        return value
    

    def psi_num(self):
        """Calculate the PSI for a single variable

        Args:
            expected_array - numpy array of original values: array
            actual_array - numpy array of new values, same size as expected: array
            buckets - number of percentile ranges to bucket the values into: int

        Returns:
            psi_value - PSI for column: float
            psi_dict - input in PSI for each bucket: dict
            new_cats - new categories
            (for not categorical data inapplicable - returns empty list): list
            abs_cats - categories that absents in actual column
            (for not categorical data inapplicable - returns empty list): list

        """
        buckets = 10
        breakpoints = np.arange(0, (buckets)/10, 0.1)
        
        # Заплатка, на случай, если в актуальной таблице появидись значения отличные от null
        if self.expected_nulls == self.expected_len and self.actual_nulls != self.actual_len:
            breakpoints = np.array(list(sorted(set(np.nanquantile(self.actual, breakpoints)))))
        else:
            breakpoints = np.array(list(sorted(set(np.nanquantile(self.expected, breakpoints)))))
        

        actual_nulls = self.actual_nulls / self.actual_len
        expected_nulls = self.expected_nulls / self.expected_len
        breakpoints = np.concatenate(([-np.inf], breakpoints, [np.inf]))
        expected_percents = np.histogram(self.expected, breakpoints)
        actual_percents = np.histogram(self.actual, breakpoints)
        #breakpoints[0] = -np.inf
        #breakpoints[-1] = np.inf
        expected_percents = [p/self.expected_len for p in expected_percents[0]]
        actual_percents = [p/self.actual_len for p in actual_percents[0]]
        if self.expected_nulls==0 and actual_nulls==expected_nulls:
            expected_percents=expected_percents
            actual_percents=actual_percents
            nulls = False
        else:
            expected_percents.append(expected_nulls)
            actual_percents.append(actual_nulls)
            nulls = True
            
        points = [i for i in breakpoints]
        intervals = [f"({np.round(points[i], 5)};{np.round(points[i+1], 5)})" for i in range(len(points)-1)]
        if nulls:
                intervals = np.append(intervals, 'empty_values')

        if self.plot:
            self.plots(nulls, expected_percents, actual_percents, breakpoints, intervals)
            
        psi_dict = {}
        for i in range(0, len(expected_percents)):
            psi_val = self.sub_psi(expected_percents[i], actual_percents[i])
            psi_dict.update({intervals[i]: psi_val})

        psi_value = np.sum(list(psi_dict.values()))
        psi_dict = {k:v for k,v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)}
        new_cats = []
        abs_cats = []

        return psi_value, psi_dict, new_cats, abs_cats
         
        
    def uniq_psi(self):
        """Counts psi for categorical unique counts > 100

        Returns:
            psi_value - PSI for column: float
            psi_dict - input in PSI for each bucket: dict
            new_cats - new categories
            (for not categorical data inapplicable - returns empty list): list
            abs_cats - categories that absents in actual column
            (for not categorical data inapplicable - returns empty list): list

        """
        actual_nulls = self.actual_nulls / self.actual_len
        expected_nulls = self.expected_nulls / self.expected_len
        actual_not_nulls_arr = self.actual[~np.isnan(self.actual)]
        expected_not_nulls_arr = self.expected[~np.isnan(self.expected)]
        actual_not_nulls = len(actual_not_nulls_arr) / self.actual_len
        expected_not_nulls = len(expected_not_nulls_arr) / self.expected_len
        expected_percents = [expected_not_nulls, expected_nulls]
        actual_percents = [actual_not_nulls, actual_nulls]
        breakpoints = ['good_data', 'nulls']
        if self.plot:
            self.plots(False, expected_percents, actual_percents, breakpoints, breakpoints)

        psi_dict = {}
        for i in range(0, len(expected_percents)):
            psi_val = self.sub_psi(expected_percents[i], actual_percents[i])
            if breakpoints[i] == 'None':
                psi_dict.update({"empty_value": psi_val})
            else:
                psi_dict.update({breakpoints[i]: psi_val})
        psi_value = np.sum(list(psi_dict.values()))
        jac_metric = self.jac()
        new_cats, abs_cats = [], []
        psi_dict = {k:v for k,v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)}
        if psi_value >= 0.2:
            psi_value = psi_value
            psi_dict.update({"metric": "stability_index"})
        else:
            psi_value = 1 - jac_metric
            psi_dict.update({"metric": "unique_index"})
        logger.info(f'PSI for categorical unique >100 is {round(psi_value, 6)}')
        return psi_value, psi_dict, new_cats, abs_cats

    
    def psi_categ(self):
        """Counts psi for categorical data exclude unique counts > 100

        Returns:
            psi_value - PSI for column: float
            psi_dict - input in PSI for each bucket: dict
            new_cats - new categories
            (for not categorical data inapplicable - returns empty list): list
            abs_cats - categories that absents in actual column
            (for not categorical data inapplicable - returns empty list): list

        """
        expected_uniq_count = len(self.expected_uniqs)
        actual_uniq_count = len(self.actual_uniqs)
        #правило для категориальных > 100
        if expected_uniq_count > 100 or actual_uniq_count > 100:
            psi_value, psi_dict, new_cats, abs_cats = self.uniq_psi()
            logger.info(f'PSI is {round(psi_value, 6)}')
            return psi_value, psi_dict, new_cats, abs_cats

        expected_dict = pd.DataFrame(self.expected, columns=[self.column_name]).groupby(self.column_name) \
            [self.column_name].count().sort_values(ascending=False).to_dict()
        actual_dict = pd.DataFrame(self.actual, columns=[self.column_name]).groupby(self.column_name) \
            [self.column_name].count().sort_values(ascending=False).to_dict()
        breakpoints = list(set(list(expected_dict.keys()) + list(actual_dict.keys())))
        new_cats = [k for k in actual_dict.keys() if k not in expected_dict.keys()]
        abs_cats = [k for k in expected_dict.keys() if k not in actual_dict.keys()]
        expected_dict_re = dict()
        actual_dict_re = dict()
        for b in breakpoints:
            if b in expected_dict and b not in actual_dict:
                expected_dict_re.update({b : expected_dict[b]})
                actual_dict_re.update({b: 0})
            elif b not in expected_dict and b in actual_dict:
                expected_dict_re.update({b : 0})
                actual_dict_re.update({b: actual_dict[b]})
            elif b in expected_dict and b in actual_dict:
                actual_dict_re.update({b: actual_dict[b]})
                expected_dict_re.update({b : expected_dict[b]})
        category_names = [c for c in expected_dict_re.keys()]
        groups = {}
        g_counts = len(category_names)
        group_num = 20
        if g_counts <= group_num:
            for g_n, val in enumerate(category_names):
                groups[val] = g_n
        else:
            group_size = np.floor(g_counts / group_num)
            current_pos = 0
            reminder = g_counts % group_num
            for g_n in range(group_num):
                if g_n < group_num - reminder:
                    group_values = category_names[int(current_pos): int(current_pos + group_size)]
                    current_pos += group_size
                else:
                    group_values = category_names[int(current_pos): int(current_pos + group_size + 1)]
                    current_pos += group_size + 1
                for val in group_values:
                    groups[val] = g_n
        group_sum = 0
        exp_dict = {}
        act_dict = {}
        group_re = -1
        cat_group_name = ""
        group_name_re = ""
        for  k, v in groups.items():
            current_group = v
            if current_group == group_re:
                group_re = v
                exp_dict.pop(group_name_re, None)
                act_dict.pop(group_name_re, None)
                cat_group_name = cat_group_name + ", " + str(k)
                group_sum_exp += expected_dict_re[k]
                group_sum_act += actual_dict_re[k]
                exp_dict.update({cat_group_name:group_sum_exp})
                act_dict.update({cat_group_name:group_sum_act})
                group_name_re = cat_group_name
            else:
                group_name_re = str(k)
                group_re = v
                cat_group_name = str(k)
                group_sum_exp = expected_dict_re[k]
                group_sum_act = actual_dict_re[k]
                exp_dict.update({cat_group_name:group_sum_exp})
                act_dict.update({cat_group_name:group_sum_act})
        expected_percents = [e / self.expected_len for e in exp_dict.values()]
        actual_percents = [a / self.actual_len for a in act_dict.values()]

        breakpoints = [e for e in exp_dict.keys()]
        
        if self.plot:
            self.plots(False, expected_percents, actual_percents, breakpoints, breakpoints)

        psi_dict = {}
        for i in range(0, len(expected_percents)):
            psi_val = self.sub_psi(expected_percents[i], actual_percents[i])
            if breakpoints[i] == 'None':
                psi_dict.update({"empty_value": psi_val})
            else:
                psi_dict.update({breakpoints[i]: psi_val})
        psi_value = np.sum(list(psi_dict.values()))
        psi_dict = {k:v for k,v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)}

        return psi_value, psi_dict, new_cats, abs_cats
    
    def psi_result(self):
        """Counts psi

        Returns:
            psi_value - PSI for column: float
            psi_dict - input in PSI for each bucket: dict
            new_cats - new categories
            (for not categorical data inapplicable - returns empty list): list
            abs_cats - categories that absents in actual column
            (for not categorical data inapplicable - returns empty list): list

        """
        if len(self.expected_shape) == 1:
            psi_values = np.empty(len(self.expected_shape))
        else:
            psi_values = np.empty(self.expected_shape[self.axis])

        for i in range(0, len(psi_values)):
            if self.column_type in [np.dtype('O')] or (self.expected_nulls == self.expected_len and self.actual_nulls == self.actual_len):
                psi_values, psi_dict, new_cats, abs_cats = self.psi_categ()
            else:
                psi_values, psi_dict, new_cats, abs_cats = self.psi_num()

        logger.info(f'PSI values: {round(psi_values,2)}')
        return round(psi_values,2), psi_dict, new_cats, abs_cats
    
    
def report(expected, actual, plot=False):
    """Func over class to create report according to the table

    Args:
        expected: spark dataframe
        actual: spark dataframe
        plot - default=False

    Returns:
        df - report in dataframe format: pd.DataFrame

    """
    logger.info('Creating report')
    assert len(expected.columns) == len(actual.columns)
    data_cols = expected.columns
    score_dict = {}
    df = pd.DataFrame()
    new_cat_dict = {}
    for col in data_cols:
        a = expected
        b = actual
        psi_res = psi(a, b, col, plot=plot)
        # отладка, в случае ошибки  выдаст прооблемный столбец
        try:
            score, psi_dict, new_cats, abs_cats = psi_res.psi_result()
        except:
            print(col)
            logger.error('')
            continue
        if len(new_cats) > 0:
            new_cat_dict.update({col:new_cats})
        score_dict.update({col: score})
        check_result = "OK" if score < 0.2 else "NOK"
        #psi_dict = {k:v for k,v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)}
        failed_buckets = list(psi_dict.keys())[:5] if score > 0.2 else []
        if 'metric' in psi_dict:
            new_cats = None
            abs_cats = None
            metric_name = psi_dict['metric']
            if metric_name == 'unique_index':
                failed_buckets = None
        else:
            metric_name = 'stability_index'
        df_tmp = pd.DataFrame({"column": col, "anomaly_score":score, 'metric_name': metric_name, 'check_result': check_result,
                                "failed_bucket":f"{failed_buckets}", 'new_category':f"{new_cats}", 'disapeared_category':f"{abs_cats}"}, index=[1])
        df = pd.concat([df, df_tmp])  
    df = df.reset_index(drop=True)
    return df