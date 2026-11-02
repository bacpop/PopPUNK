import pandas as pd

from PopPUNK.utils import isolateNameToLabel


def test_isolate_name_to_label_accepts_ints():
    labels = isolateNameToLabel([2025133, 7, 'dir/sample.1'])
    assert labels == ['2025133', '7', 'sample_1']


def test_info_csv_numeric_ids_keep_leading_zeros(tmp_path):
    info = tmp_path / 'info.csv'
    info.write_text('ID,location\n007,Taiwan\n2025133,Japan\n')
    epi = pd.read_csv(info, index_col=False, quotechar='"', dtype=str)
    assert isolateNameToLabel(epi.iloc[:, 0]) == ['007', '2025133']
