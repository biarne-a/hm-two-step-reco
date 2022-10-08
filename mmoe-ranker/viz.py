import pandas as pd
import sweetviz as sv

from features import Features


if __name__ == '__main__':
    train_df = pd.read_csv('train_df.csv')
    for categ_variable in Features.ALL_CATEG_FEATURES:
        train_df[categ_variable] = train_df[categ_variable].astype(str)
    my_report = sv.analyze(train_df, target_feat=Features.LABEL1)
    output_html = 'sweetviz_report.html'
    my_report.show_html(output_html)
