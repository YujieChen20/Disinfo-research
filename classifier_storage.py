import pandas as pd
import datetime
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
from classifier_helper import getLinkDataRestrained as get_link_data_restrained # TODO: deprecate
from classifier_helper import psi as link_energy
from classifier_helper import computeH as compute_energy_graph
from classifier_helper import compute_bot_probabilities
from classifier_helper import fmt_pct
from classifier_helper import fmt_n
from app.gcs_service import GoogleCloudStorageService
from app import server_sleep
from dotenv import load_dotenv


load_dotenv()
# Choose dataset
DATASET = os.getenv("DATASET", default="Impeachment_2021")


# Generate Date
dateList = []
start_dt = os.getenv("START_DATE", default="2020-11-15")
end_dt = os.getenv("END_DATE", default="2021-03-20")
start_date = datetime.datetime.strptime(start_dt, "%Y-%m-%d")
end_date = datetime.datetime.strptime(end_dt, "%Y-%m-%d")
delta = datetime.timedelta(days=1)
while start_date <= end_date:
    dateList.append(start_date.strftime("%Y-%m-%d"))
    start_date += delta


# Classifier
# https://github.com/s2t2/tweet-analysis-2020/blob/429b9b62b2eea23183d74f87c7376bd2b9832ca3/app/botcode_v2/classifier.py#L7
class NetworkClassifier:
    def __init__(self, rt_graph, weight_attr="weight", mu=1, alpha_percentile=0.999, lambda_00=0.61, lambda_11=0.83):
        """
        Takes all nodes in a retweet graph and assigns each user a score from 0 (human) to 1 (bot).
        Then writes the results to CSV file.
        """
        self.rt_graph = rt_graph
        self.weight_attr = weight_attr

        # PARAMS FOR THE LINK ENERGY FUNCTION...
        self.mu = mu
        self.alpha_percentile = alpha_percentile
        self.lambda_00 = lambda_00
        self.lambda_11 = lambda_11
        self.epsilon = 10**(-3) #> 0.001
        #self.lambda_01 = 1
        #self.lambda_10 = self.lambda_00 + self.lambda_11 - self.lambda_01 + self.epsilon

        # ARTIFACTS OF THE BOT CLASSIFICATION PROCESS...
        self.energy_graph = None
        self.bot_ids = None
        self.user_data = None

    @property
    def links(self):
        print("-----------------")
        print("LINKS...")
        return get_link_data_restrained(self.rt_graph, weight_attr=self.weight_attr)

    @property
    def in_degrees(self):
        return self.rt_graph.in_degree(weight=self.weight_attr)

    @property
    def out_degrees(self):
        return self.rt_graph.out_degree(weight=self.weight_attr)

    @property
    def alpha(self):
        """Params for the link_energy function"""
        in_degrees_list = [v for _, v in self.in_degrees]
        out_degrees_list = [v for _, v in self.out_degrees]
        #print("MAX IN:", fmt_n(max(in_degrees_list)))  # > 76,617
        #print("MAX OUT:", fmt_n(max(out_degrees_list)))  # > 5,608

        alpha_in = np.quantile(in_degrees_list, self.alpha_percentile)
        alpha_out = np.quantile(out_degrees_list, self.alpha_percentile)
        #print("ALPHA IN:", fmt_n(alpha_in))  # > 2,252
        #print("ALPHA OUT:", fmt_n(alpha_out))  # > 1,339

        return [self.mu, alpha_out, alpha_in]

    @property
    def link_energies(self):
        """TODO: refactor by looping through the edges in the RT graph instead....
            link[0] is the edge[0]
            link[1] is the edge[1]
            link[4] is the weight attr value
        """
        print("-----------------")
        print("ENERGIES...")
        return [(
            link[0],
            link[1],
            link_energy(
                link[0], link[1], link[4],
                self.in_degrees, self.out_degrees,
                self.alpha, self.lambda_00, self.lambda_11, self.epsilon
            )
        ) for link in self.links]

    @property
    def prior_probabilities(self):
        return dict.fromkeys(list(self.rt_graph.nodes), 0.5)  # set all screen names to 0.5

    def compile_energy_graph(self):
        print("COMPILING ENERGY GRAPH...")
        self.energy_graph, self.bot_ids, self.user_data = compute_energy_graph(self.rt_graph, self.prior_probabilities,
                                                                               self.link_energies, self.out_degrees,
                                                                               self.in_degrees)
        # self.human_names = list(set(self.rt_graph.nodes()) - set(self.bot_ids))
        print("-----------------")
        print("ENERGY GRAPH:", type(self.energy_graph))
        print("NODE COUNT:", fmt_n(self.energy_graph.number_of_nodes()))
        print(
            f"BOT COUNT: {fmt_n(len(self.bot_ids))} ({fmt_pct(len(self.bot_ids) / self.energy_graph.number_of_nodes())})")
        print("USER DATA:", fmt_n(len(self.user_data.keys())))

    @property
    def bot_probabilities(self):
        if not self.energy_graph and not self.bot_ids:
            self.compile_energy_graph()

        return compute_bot_probabilities(self.rt_graph, self.energy_graph, self.bot_ids)

    @property
    def bot_probabilities_df(self):
        df = pd.DataFrame(list(self.bot_probabilities.items()), columns=["user_id", "bot_probability"])
        df.index.name = "row_id"
        df.index = df.index + 1
        print("--------------------------")
        print("CLASSIFICATION COMPLETE!")
        print(df.head())
        print("... < 50% (NOT BOTS):", fmt_n(len(df[df["bot_probability"] < 0.5])))
        print("... = 50% (NOT BOTS):", fmt_n(len(df[df["bot_probability"] == 0.5])))
        print("... > 50% (MAYBE BOTS):", fmt_n(len(df[df["bot_probability"] > 0.5])))
        print("... > 90% (LIKELY BOTS):", fmt_n(len(df[df["bot_probability"] > 0.9])))
        return df

    def generate_bot_probabilities_histogram(self, img_filepath=None,
                                             title="Bot Probability Scores (excludes 0.5)"):
        probabilities = self.bot_probabilities_df["bot_probability"]
        num_bins = round(len(probabilities) / 10)
        counts, bin_edges = np.histogram(probabilities,
                                         bins=num_bins)  # ,normed=True #> "VisibleDeprecationWarning: Passing `normed=True` on non-uniform bins has always been broken"...
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        plt.plot(bin_edges[1:], cdf / cdf[-1])
        plt.grid()
        plt.xlabel("Bot probability")
        plt.ylabel("CDF")

        plt.hist(probabilities[probabilities < 0.5])
        plt.hist(probabilities[probabilities > 0.5])
        plt.grid()
        plt.xlabel("Bot probability")
        plt.ylabel("Frequency")
        plt.title(title)

        plt.savefig(img_filepath)

        #if show_img:
            #plt.show()

gcs_service = GoogleCloudStorageService()

# Load Retweet graphs
for date in dateList:
    fileName = "graph_" + str(date) + ".gpickle"
    remote_graph_filePath = os.path.join(DATASET, 'RT_graph', fileName)
    local_graph_filePath = os.path.join(os.path.dirname(__file__), DATASET, "download", fileName)
    gcs_service.download(remote_graph_filePath, local_graph_filePath)
    #drive_path = '/content/drive/MyDrive/Disinfo Research Shared 2022/users/yc986/Impeachment_2021/'
    #drive_graph_filepath = os.path.join(drive_path, "RT_graph", fileName)
    rt_graph = nx.read_gpickle(local_graph_filePath)
    classifier = NetworkClassifier(rt_graph)
    df = classifier.bot_probabilities_df
    fileName_csv = "probabilities_" + str(date) + ".csv"
    fileName_png = "histogram_" + str(date) + ".png"

    csv_filepath = os.path.join(os.path.dirname(__file__), DATASET, "bot_classification", "probabilities", fileName_csv)
    img_filepath = os.path.join(os.path.dirname(__file__), DATASET, "bot_classification", "probabilities_histogram", fileName_png)

    df.to_csv(csv_filepath)
    classifier.generate_bot_probabilities_histogram(img_filepath=img_filepath)

    remote_csv_filepath = os.path.join(DATASET, 'bot_classification', "classification_df", fileName_csv)
    remote_img_filepath = os.path.join(DATASET, 'bot_classification', "classification_hist", fileName_png)
    gcs_service.upload(csv_filepath, remote_csv_filepath)
    gcs_service.upload(img_filepath, remote_img_filepath)

    server_sleep()
