# oneNeuron
oneNeuron | Perceptron

## commands used - 

```bash
git add . && git commit -m "first commit" && git push origin main
```

## Add a url -
[Git handbook](https://guides.github.com/introduction/git-handbook/)

## Add an image -
![Sample output image](plots/or.png)

## dataset -
x1|x2|y
-|-|-
0|0|0
0|1|1
1|0|1
1|1|1

## python code
```python
    def prepare_data(df):
    """it is used to generate the dependent and indenpendent vaariables

    Args:
        df (pd.DataFrame): it is a pandas dataframe

    Returns:
        tuple: returns both dependent and independent variables
    """
    logging.info("Started preparing the data into dependent and independent variables")
    X = df.drop("y", axis =1)
    y = df["y"]
    return X,y
``` 