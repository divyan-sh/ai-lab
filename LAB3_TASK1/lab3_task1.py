import pandas as pd

# Load your dataset
df_sg = pd.read_excel("./smart_grid.xlsx")

import bnlearn as bn

# Structure learning
model = bn.structure_learning.fit(df_sg)

# Learn parameters
model_update = bn.parameter_learning.fit(model, df_sg, methodtype="bayes")

# Inference 1
q1 = bn.inference.fit(model_update, variables=['Outage_Duration'], evidence={'Time': 'Morning', 'Demand_Factor': 'Medium'})

# Inference 2
q2 = bn.inference.fit(model_update, variables=['Demand_Factor'], evidence={'Overload': 'Yes', 'Weather': 'Cold'})

# Inference 3
q3 = bn.inference.fit(model_update, variables=['Number_of_Customers'], evidence={'Demand_Factor': 'High'})
