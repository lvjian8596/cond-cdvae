import click
import pandas as pd


@click.command()
@click.argument("dataset")
@click.option("-f", "--formula", help="query formula")
@click.option("-i", "--material_id", help="query material_id")
def search(dataset, formula=None, material_id=None):
    click.echo(dataset)
    ds = pd.read_feather(dataset)
    if formula is not None:
        click.echo(ds[ds.formula == formula])
    if material_id is not None:
        click.echo(ds[ds.material_id == material_id])


if __name__ == "__main__":
    search()
