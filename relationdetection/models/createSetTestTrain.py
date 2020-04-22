import pandas as pd
import click
import json

@click.command()
@click.argument(
    '--feats_file', 
    type=click.File('rb'),
    help = 'Path where to find the .csv with the features for our model.')

def create_sets(feats_file):
    """
    To be done once, to have the indexes files
    """
    df = pd.read_csv(feats_file)
    df.fillna(value="", inplace=True)
    df.set_index(["entity1","entity2", "original_sentence"], inplace=True)

        #chose the set of test and train
    df0 = df[df.relation == ""]
    df1 = df[df.relation!='']

    num_train0 = int(df0.shape[0]/2)
    num_train1 = int(df1.shape[0]/2)
    num_train0, num_train1

    df0_train = df0.sample(num_train0, random_state =7)
    df1_train = df1.sample(num_train1, random_state =7)

    df0_test = df0.drop(df0_train.index.tolist(), axis =0)
    df1_test = df1.drop(df1_train.index.tolist(), axis =0)

     #final dfs
    df_train = pd.concat([df0_train, df1_train])
    df_test = pd.concat([df0_test, df1_test])

    json.dump(df_test.id_entities_pair.tolist(), open( "indexes_test.json", 'w' ) )
    json.dump(df_train.id_entities_pair.tolist(), open( "indexes_train.json", 'w' ) )

if __name__ == "__main__":
    create_sets()