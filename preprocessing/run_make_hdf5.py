"""
Making the hdf5 datasets.

"""

if __name__ == "__main__":
    
    from pathlib import Path
    import make_hdf5

    
    base_path = Path('.\\Data_070119\\')
    max_xyn = (236, 236)
    
    h5ck = make_hdf5.DatasetMaker(max_xyn)
    data, GTVtarget = h5ck.get_data(base_path, MRI=True, aug=True, remove_slices=True)
    
#    [train_slices, val_slices, test_slices] = h5ck.train_val_test_slices()
#    [train_ids, val_ids, test_ids] = h5ck.train_val_test_patients()
#   
    
    init = make_hdf5.HDFMaker(data=data, 
                      filename='data_070119_PETCT_final.h5',
                      target=GTVtarget)
    init.make_hdf(slices=h5ck.train_val_test_slices(), 
                  ids=h5ck.train_val_test_patients())
    
    