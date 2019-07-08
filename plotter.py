import matplotlib.pyplot as plt
import numpy as np

x_label='x label'
y_label='y label'
plot_title="Plot"

def show_one(list):

	plt.plot(list)

	plt.xlabel(x_label)
	plt.ylabel(y_label)


	plt.legend()

	plt.show()

def show_multi(list_of_lists):
	for i in list_of_lists:
		plt.plot(i)

	plt.xlabel(x_label)
	plt.ylabel(y_label)


	plt.legend()

	plt.show()