# This script uses the Snowstorm SNOMED-CT API. A standardised FHIR API is also available.

# Note that we recommend running your own Snowstorm instance for heavy script use.
# See https://github.com/IHTSDO/snowstorm

from urllib.request import urlopen, Request
from urllib.parse import quote
import json

baseUrl = 'http://localhost:8080'
edition = 'MAIN'
version = '2025-08-01'
user_agent = 'Python'

snomed_ct_top_level_key_to_id = {
    'body structure': 123037004,
    'finding': 404684003,
    'environment / location': 308916002,
    'event': 272379006,
    'observable entity': 363787002,
    'organism': 410607006,
    'product': 373873005,
    'physical force': 78621006,
    'physical object': 260787004,
    'procedure': 71388002,
    'qualifier value': 362981000,
    'record artifact': 419891008,
    'situation': 243796009,
    'metadata': 900000000000441003,
    'social context': 48176007,
    'special concept': 370115009,
    'specimen': 123038009,
    'staging scale': 254291000,
    'substance': 105590001
}

def _branch_path(branch: str | None = None) -> str:
    """生成 {edition}/{version} 或自定义分支路径"""
    if branch:
        return branch
    if version:
        return f"{edition}/{version}"
    return edition

def _urlopen_with_header(url):
    # adds User-Agent header otherwise urlopen on its own gets an IP blocked response
    req = Request(url)
    req.add_header('User-Agent', user_agent)
    return urlopen(req)

#Prints fsn of a concept
def getConceptById(id):
    url = baseUrl + '/browser/' + edition + '/' + version + '/concepts/' + id
    response = _urlopen_with_header(url).read()
    data = json.loads(response.decode('utf-8'))

    print (data['fsn']['term'])

#Prints description by id
def getDescriptionById(id):
    url = baseUrl + '/' + edition + '/' + version + '/descriptions/' + id
    response = _urlopen_with_header(url).read()
    data = json.loads(response.decode('utf-8'))

    print (data['term'])

#Prints number of descriptions containing the search term with a specific semantic tag
def getDescriptionsByStringFromProcedure(searchTerm, semanticTag):
    url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?term=' + quote(searchTerm) + '&conceptActive=true&semanticTag=' + quote(semanticTag) + '&groupByConcept=false&searchMode=STANDARD&offset=0&limit=50'
    response = _urlopen_with_header(url).read()
    data = json.loads(response.decode('utf-8'))

    print (data['totalElements'])
    
 #Prints snomed code for searched disease or symptom
def getSnomedCodeSimilar(searchTerm):
    url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?term=' + quote(searchTerm) + '&conceptActive=true&groupByConcept=false&searchMode=STANDARD&offset=0&limit=50'
    response = _urlopen_with_header(url).read()
    data = json.loads(response.decode('utf-8'))

    for term in data['items']:
      if searchTerm in term['term']:
        print("{} : {}".format(term['term'], term['concept']['conceptId']))
 
def getSnomedCode(searchTerm):
    url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?term=' + quote(searchTerm) + '&conceptActive=true&groupByConcept=false&searchMode=STANDARD&offset=0&limit=50'
    response = _urlopen_with_header(url).read()
    data = json.loads(response.decode('utf-8'))

    for term in data['items']:
      if searchTerm == term['term']:
        print("{} : {}".format(term['term'], term['concept']['conceptId']))

#Prints number of concepts with descriptions containing the search term
# curl --silent "http://localhost:8080/browser/MAIN/descriptions?&limit=50&term=burn&conceptActive=true&lang=english&skipTo=0&returnLimit=50"
def getDescriptionByString(
        searchTerm: str,
        branch: str | None = None
    ):
    """
    使用 /browser/{branch}/descriptions 搜索，返回：
        - concept_ids: 去重后的 conceptId 列表
        - items:      每条命中的精简信息（含 term、concept 概要等）
    支持分页（offset 风格）；max_pages 控制拉取页数。
    """
    branch_path = _branch_path(branch)

    # url = baseUrl + '/browser/' + 'MAIN' + '/' + 'concepts?term=' + quote(searchTerm) + '&activeFilter=true&offset=0&limit=50'
    url_origin = baseUrl + '/browser/' + 'MAIN' + '/' + 'descriptions?&limit=50&term=' + quote(searchTerm) + '&conceptActive=true&lang=english&skipTo=0&returnLimit=50'

    url = (
        f"{baseUrl}/browser/{branch_path}/descriptions"
        f"?&term={quote(searchTerm)}"
        f"&conceptActive=true"
        f"&lang=english"
        f"&returnLimit={50}"
    )

    response = _urlopen_with_header(url).read()
    data = json.loads(response.decode('utf-8'))
    if data['items']:
        return data['items']
    else:
        return []

# GET /browser/{branch}/concepts/{id}/ancestors?form=inferred
def get_ancestors(
    concept_id: str,
    branch: str | None = None,
    form: str = "inferred"
):
    """
    官方推荐：GET /{branch}/concepts/{id}/ancestors
    - relationships 不会包含；通过 form 控制 stated/inferred
    """
    branch_path = _branch_path(branch)
    url = f"{baseUrl}/browser/{branch_path}/concepts/{concept_id}/ancestors?form={quote(form)}"
    data = json.loads(_urlopen_with_header(url).read().decode("utf-8"))
    return data

def get_concept_detail(
    concept_id: str,
    *,
    branch: str | None = None,
    form: str = "inferred",
    include_descriptions: bool = False
):
    """
    官方推荐：GET /{branch}/concepts/{id}
    - relationships 默认会包含；通过 form 控制 stated/inferred
    - include_descriptions=True 时附带 descriptions
    """
    branch_path = _branch_path(branch)
    params = [f"form={quote(form)}"]
    if include_descriptions:
        params.append("includeDescriptions=true")
    url = f"{baseUrl}/{branch_path}/concepts/{concept_id}"
    if params:
        url += "?" + "&".join(params)

    data = json.loads(_urlopen_with_header(url).read().decode("utf-8"))
    return data

def get_concepts_details(
    concept_ids: list[str],
    *,
    branch: str | None = None,
    form: str = "inferred",
    include_descriptions: bool = False
):
    """
    遍历概念 id 列表逐个取详情（简洁可靠；也便于失败重试/断点续拉）
    """
    details = []
    for cid in concept_ids:
        details.append(
            get_concept_detail(
                cid,
                branch=branch,
                form=form,
                include_descriptions=include_descriptions
            )
        )
    return details

def find_top_level_category(concept_id: str) -> str | None:
    """
    追溯到顶层类别（直接父节点是顶层类别）
    """
    snomed_ct_top_level_id_to_key = {str(v): k for k, v in snomed_ct_top_level_key_to_id.items()}
    ancestors = get_ancestors(concept_id)
    ancestor_ids = {str(item['conceptId']) for item in ancestors}
    for ancestor_id in ancestor_ids:
        if ancestor_id in snomed_ct_top_level_id_to_key.keys():
            return snomed_ct_top_level_id_to_key[ancestor_id]
    return ""


if __name__ == "__main__":

    result = getDescriptionByString('Burn', '')

    concept_id = 125666000
    ancestors = get_ancestors(str(concept_id), '', '')

    with open('utils/examples.json', 'w', encoding='utf-8') as f:
        json.dump(ancestors, f, ensure_ascii=False, indent=4)
    print(f"Ancestors of {concept_id} saved to examples.json")

    top_level_category = find_top_level_category(str(concept_id))
    print(top_level_category)


