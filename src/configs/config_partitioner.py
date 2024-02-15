from ingester3.ViewsMonth import ViewsMonth

def get_partitioner_dict(partion):

    """Returns the partitioner_dict for the given partion."""

    if partion == 'calibration':

        partitioner_dict = {"train":(121,444),"predict":(445,492)} 

    if partion == 'testing':

        partitioner_dict = {"train":(121,444),"predict":(445,492)} 

    if partion == 'forecasting':

        last_month =  ViewsMonth.now().id - 2

        partitioner_dict = {"train":(121, last_month),"predict":(last_month, last_month + 36)} # 36 should not be hard coded...  

    return partitioner_dict    


