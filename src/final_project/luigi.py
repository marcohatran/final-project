import pandas as pd
from luigi import ExternalTask, Task

from final_project.target import *
from final_project.task import *


class BillingData(ExternalTask):
    """An ExternalTask that returns a CSVTarget from
    an AWS Source, __version__ was manually specified
    to work with get_salted_version properly. In real
    life cases class __version__ needs to be updated
    every time the class is rerun."""
    __version__ = '1.0'

    def output(self):
        return CSVTarget('s3://########/Billing/', glob='*', flag=None)


class FairHealthData(ExternalTask):
    def output(self):
        return LocalTarget('fair_health.xlsx')


class FairHealth(Task):

    def requires(self):
        return FairHealthData()

    def run(self):
        df2 = pd.read_excel(self.input().open('r'), header=[0])
        df2 = df2.drop(columns=['Modifier', 'Unnamed: 6', 'Year', 'Charge'])
        df2 = df2[df2['Fair Health'].notnull()]
        df2.reset_index(drop=True)

    def output(self):
        return LocalTarget('clean_fairhealth.csv')


class CleanedBilling(Task):
    """A Luigi task which relies on two ExternalTasks, BillingData
     and FairHealth in order to read a Dask object and clean it up
     Again, __version__ is manually specified.

    rtype: Parquet file with a salted name according to
           SaltedOutput"""

    __version__ = '1.0'

    requires = Requires()
    billing_data, fair_health = Requirement(BillingData, FairHealth)
    output = SaltedOutput("Dataset.csv/", target_class=CSVTarget, glob='*.csv')

    def run(self):

        dsk = self.input()['billing_data'].read_dask(check_complete=False, parse_dates=['Received Date'])
        df2 = self.input()['fair_health'].read_excel()

        dsk = dsk.merge(df2.set_index('CPT Code'), on='CPT Code')
        out = dsk.set_index('CPT Code')
        self.output().write_dask(out, compression='gzip')


class LinearRegression(Task):

    requires = Requires()
    cleaned_reviews = Requirement(CleanedBilling)
    output = SaltedOutput("Dataset.csv/", target_class=CSVTarget, glob='*.csv')

    def train(partition):
        est = LinearRegression()
        est.fit(partition[['Charged', 'Allowed - EFLab']].values, partition.Collected.values)
        return est

    def run(self):
        dsk = self.input()['Dataset'].read_dask()
        dsk.groupby('Inurance Name').apply(self.train, meta=object).compute().sort_index()
