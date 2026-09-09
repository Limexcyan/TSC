from json import dumps
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from rdkit.Chem.fmcs import fmcs
from rdkit.Chem import MolFromSmiles, MolFromSmarts
from rdkit.Chem.Draw import MolToSVG

import logging

EMBED_DISPLAY = """<div>
View cluster: <select id="cluster-picker" name="cluster-picker" class="material-select">
<option value="0" selected="selected">0</option>
</select>
<div style="min-height:80vh;">
<div>
<h1>Most common substructures:</h1>
<div id="cluster-mcs-view" style=""width:100%;display:grid;grid-template-columns: 30% 30% 30%;margin-bottom:3em;">
<div><h3>Within top 10 compounds:</h3><div id="cluster-mcs-view-10" style="border-radius:50px;overflow:hidden;height:300px"></div></div>
<div><h3>Within top 25 compounds:</h3><div id="cluster-mcs-view-25" style="border-radius:50px;overflow:hidden;height:300px"></div></div>
<div><h3>Within top 50 compounds:</h3><div id="cluster-mcs-view-50" style="border-radius:50px;overflow:hidden;height:300px"></div></div>
</div>
</div>
<div>
<h1>Top 50 compounds in this cluster:</h1>
<div id="cluster-top" style="overflow:scroll;max-height:80vh;">
</div>
</div>
</div>
</div>
"""

EMBED_SCRIPT = """<script>
    const task_data = JSON.parse(document.getElementById("analysis-data").innerText);
    const picker = document.getElementById("cluster-picker");
    for (var x in task_data.slice(1)) {
        picker.options[picker.options.length] = (new Option(x, x));
    }
    function __build_list(node, records, records_svg) {
        for (var i = 0; i < records_svg.length; i++) {
            let new_node = document.createElement("div");
            new_node.classList = "margin:0;padding:0;display:inline-flex;flex-direction:row;justify-content:space-between;height=150px;overflow:hidden;width:80%"
            let text_node = document.createElement("div");
            text_node.innerHTML = `<h3>${i}.</h3>`;
            var title = document.createElement("h4");
            var smiles = document.createElement("h5");
            var distance = document.createElement("h5");
            var image = document.createElement("div");
            title.innerText = records["chembl_id"][i];
            smiles.innerText = records["smiles"][i];
            distance.innerText = records["distance"][i];
            image.innerHTML = records_svg[i];
            image.classList = "border-radius:50px;overflow:hidden;height:200px"
            text_node.appendChild(title);
            text_node.appendChild(smiles);
            text_node.appendChild(distance);
            new_node.appendChild(text_node);
            new_node.appendChild(image);
            node.appendChild(new_node);
        }
    }
    picker.onchange = function () {
        let data = JSON.parse(document.getElementById("analysis-data").innerText);
        let side_list = document.getElementById("cluster-top");
        document.getElementById("cluster-mcs-view-10").innerHTML = data[this.value]["top10mcs_svg"];
        document.getElementById("cluster-mcs-view-25").innerHTML = data[this.value]["top25mcs_svg"];
        document.getElementById("cluster-mcs-view-50").innerHTML = data[this.value]["top50mcs_svg"];
        side_list.innerHTML = "";
        __build_list(side_list, data[this.value]["top50"], data[this.value]["top50_svg"]);
    }
</script>
"""
logger = logging.getLogger(__name__)

def mcs_analysis(input_file: Path, output_file: Path):
    data = pl.read_csv(input_file)
    cluster_count = data.get_column("centroid").n_unique()

    result_data = []
    for centroid in range(cluster_count):
        centroid_data = data.filter(pl.col("centroid") == centroid)
        centroid_result = {
            "id": centroid,
            "top50": centroid_data.top_k(50,by="distance").to_dict(as_series=False),
            "top50_svg": centroid_data.top_k(50,by="distance").get_column("smiles").map_elements(lambda smiles: MolToSVG(MolFromSmiles(smiles),size=(200, 200))).to_list(),
            "top10mcs_svg": MolToSVG(MolFromSmarts(fmcs(centroid_data.top_k(10,by="distance").get_column("smiles").map_elements(MolFromSmiles)).smarts)),
            "top25mcs_svg": MolToSVG(MolFromSmarts(fmcs(centroid_data.top_k(25,by="distance").get_column("smiles").map_elements(MolFromSmiles)).smarts)),
            "top50mcs_svg": MolToSVG(MolFromSmarts(fmcs(centroid_data.top_k(50,by="distance").get_column("smiles").map_elements(MolFromSmiles)).smarts)),
        }
        result_data.append(centroid_result)
        logger.info("finished mcs on cluster %d/%d",centroid,cluster_count - 1)

    with open(output_file, "w") as fo:
        logger.info("embedding mcs results in html")
        fo.write('<script id="analysis-data" type="application/json">')
        fo.write(dumps(result_data))
        fo.write("</script>")
        fo.write(EMBED_DISPLAY)
        fo.write(EMBED_SCRIPT)
    return
