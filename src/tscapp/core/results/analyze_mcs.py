from json import dumps
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem import MolFromSmiles, MolFromSmarts
from rdkit.Chem.Draw import MolToSVG

import logging

EMBED_DISPLAY = """<div>
<div style="width:20vh;">
View cluster: <select id="cluster-picker" name="cluster-picker" class="material-select">
<option value="0" selected="selected">id</option>
</select>
</div>
<div style="min-height:80vh;">
<div>
<h1>Most common substructures:</h1>
<div id="cluster-mcs-view" style="width:100%;display:grid;grid-template-columns: 30% 30% 30%;margin-bottom:3em;">
<div><h3>Within top 10 compounds:</h3><div id="cluster-mcs-view-10" style="border-radius:50px;overflow:hidden;height:300px;width:300px"></div></div>
<div><h3>Within top 25 compounds:</h3><div id="cluster-mcs-view-25" style="border-radius:50px;overflow:hidden;height:300px;width:300px"></div></div>
<div><h3>Within top 50 compounds:</h3><div id="cluster-mcs-view-50" style="border-radius:50px;overflow:hidden;height:300px;width:300px"></div></div>
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
    for (var x in task_data) {
        picker.options[picker.options.length] = (new Option(x, x));
    }
    function __build_list(node, records, records_svg) {
        for (var i = 0; i < records_svg.length; i++) {
            let new_node = document.createElement("div");
            new_node.style.margin = 0;
            new_node.style.padding = 0;
            new_node.style.display = "inline-flex";
            new_node.style.flexDirection = "row";
            new_node.style.justifyContent = "space-between";
            new_node.style.height = "200px";
            new_node.style.marginBottom = "2em";
            new_node.style.width = "90%";
            let text_node = document.createElement("div");
            text_node.style.width = "calc(100% - 250px)";
            text_node.innerHTML = `<h3>${i+1}.</h3>`;
            var title = document.createElement("h4");
            var smiles = document.createElement("h5");
            var distance = document.createElement("h5");
            var image = document.createElement("div");
            title.innerText = `${records["filename"][i]} - ${records["chembl_id"][i]}`;
            smiles.innerText = records["smiles"][i];
            distance.innerText = records["distance"][i];
            image.innerHTML = records_svg[i];
            image.style.borderRadius = "50px";
            image.style.overflow = "hidden";
            image.style.height = "200px";
            image.style.width = "200px";
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
            "top50": centroid_data.top_k(50,by="distance",reverse=True).to_dict(as_series=False),
            "top50_svg": centroid_data.top_k(50,by="distance",reverse=True).get_column("smiles").map_elements(lambda smiles: MolToSVG(MolFromSmiles(smiles),size=(200, 200))).to_list(),
            "top10mcs_svg": MolToSVG(MolFromSmarts(FindMCS(centroid_data.top_k(10,by="distance",reverse=True).get_column("smiles").map_elements(MolFromSmiles)).smartsString)),
            "top25mcs_svg": MolToSVG(MolFromSmarts(FindMCS(centroid_data.top_k(25,by="distance",reverse=True).get_column("smiles").map_elements(MolFromSmiles)).smartsString)),
            "top50mcs_svg": MolToSVG(MolFromSmarts(FindMCS(centroid_data.top_k(50,by="distance",reverse=True).get_column("smiles").map_elements(MolFromSmiles)).smartsString)),
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
