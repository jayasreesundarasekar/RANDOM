import streamlit as st
import pandas as pd
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw
import py3Dmol
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile, os

st.set_page_config(page_title="Molecule Studio | Cheminformatics Lab", page_icon="⚛️", layout="wide", initial_sidebar_state="expanded")

# ---------- visual system ----------
st.markdown(r'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#eaf4ff;--muted:#91a8c4;--panel:#0e1b2e;--line:rgba(148,180,219,.16);--cyan:#65e7df;--blue:#76a9ff;--violet:#a994ff;}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(ellipse at 80% 0%,rgba(52,112,164,.19),transparent 35%),radial-gradient(ellipse at 5% 35%,rgba(70,57,139,.13),transparent 30%),#07111f;color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c192b,#091321);border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:#dceaff}
.block-container{padding-top:1.6rem;max-width:1450px}
h1,h2,h3{letter-spacing:-.04em!important;color:#f2f8ff!important}
.eyebrow{font:500 11px 'DM Mono',monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);margin-bottom:10px}
.hero{position:relative;overflow:hidden;border:1px solid rgba(124,185,227,.22);border-radius:26px;padding:36px 38px;background:linear-gradient(125deg,rgba(20,43,67,.97),rgba(12,23,42,.95) 62%,rgba(30,27,67,.8));min-height:280px}
.hero:after{content:' ';position:absolute;width:270px;height:270px;right:4%;top:-65px;border-radius:50%;border:1px solid rgba(101,231,223,.22);box-shadow:0 0 0 28px rgba(101,231,223,.035),0 0 0 58px rgba(118,169,255,.035),inset 0 0 55px rgba(101,231,223,.06);pointer-events:none}
.hero h1{font-size:clamp(2.1rem,4vw,3.6rem);line-height:1.03;margin:0 0 16px;max-width:720px}
.hero p{color:#a8bdd5;max-width:610px;font-size:1rem;line-height:1.8}
.pill{display:inline-block;border:1px solid rgba(101,231,223,.32);background:rgba(101,231,223,.08);color:var(--cyan);border-radius:999px;padding:7px 12px;font:11px 'DM Mono',monospace;letter-spacing:.06em;margin-bottom:18px}
.panel{border:1px solid var(--line);background:linear-gradient(145deg,rgba(17,34,55,.9),rgba(12,25,42,.86));border-radius:18px;padding:20px 22px;margin-bottom:12px}
.metric{border:1px solid var(--line);border-radius:16px;padding:17px 18px;background:linear-gradient(145deg,rgba(19,40,63,.85),rgba(13,25,42,.8));min-height:108px}
.metric .label{font:11px 'DM Mono',monospace;color:#91a8c4;text-transform:uppercase;letter-spacing:.08em}.metric .value{font-size:1.65rem;font-weight:800;color:#eff8ff;margin-top:8px}.metric .unit{font-size:.8rem;color:#8faac7;font-weight:500}
.section-kicker{font:11px 'DM Mono',monospace;color:var(--cyan);letter-spacing:.14em;text-transform:uppercase;margin:12px 0 6px}
.small-note{font-size:12px;color:#91a8c4;line-height:1.7}
div[data-testid="stVerticalBlock"]>div:has(>.element-container .hero){gap:.2rem}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid rgba(101,231,223,.28);background:linear-gradient(120deg,#4dd7d1,#78b3ff);color:#071525;font-weight:800;padding:.65rem 1rem;transition:transform .2s,filter .2s}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.08);transform:translateY(-1px);border-color:var(--cyan);color:#071525}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div{background:#0a1728!important;color:#eaf4ff!important;border-color:rgba(145,177,215,.25)!important;border-radius:10px!important}
[data-testid="stMetric"]{background:#102139;border:1px solid var(--line);padding:15px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#91a8c4!important}[data-testid="stMetricValue"]{color:#f2f8ff!important}
hr{border-color:var(--line)!important}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent}.stTabs [data-baseweb="tab"]{border:1px solid var(--line);border-radius:10px;background:#0e1b2e;padding:9px 15px;color:#a8bdd5}.stTabs [aria-selected="true"]{background:#163044!important;color:var(--cyan)!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
footer{visibility:hidden}
</style>''', unsafe_allow_html=True)

EXAMPLES={
    "Ethanol":{"smiles":"CCO","cid":702,"note":"A simple alcohol used as a solvent and fuel."},
    "Benzene":{"smiles":"c1ccccc1","cid":241,"note":"An aromatic ring and a foundational organic chemistry structure."},
    "Phenol":{"smiles":"Oc1ccccc1","cid":996,"note":"An aromatic compound bearing a hydroxyl group."},
    "Acetic acid":{"smiles":"CC(=O)O","cid":176,"note":"The characteristic acid in vinegar."},
    "Aspirin":{"smiles":"CC(=O)Oc1ccccc1C(=O)O","cid":2244,"note":"A widely known analgesic and anti-inflammatory molecule."},
    "Caffeine":{"smiles":"Cn1c(=O)c2c(ncn2C)n(C)c1=O","cid":2519,"note":"A naturally occurring stimulant found in coffee and tea."},
    "Glucose":{"smiles":"C(C1C(C(C(C(O1)O)O)O)O)O","cid":5793,"note":"A central monosaccharide in cellular energy metabolism."},
    "Lactic acid":{"smiles":"CC(O)C(=O)O","cid":612,"note":"An alpha-hydroxy acid associated with fermentation and metabolism."}
}

PROPERTY_HELP={
"Molecular Weight":"Average molecular mass, in g/mol.","Exact Molecular Weight":"Mass calculated from the most abundant isotopes, in Da.","LogP":"Estimated octanol/water partition coefficient; higher values indicate greater lipophilicity.","TPSA":"Topological polar surface area, a descriptor of molecular polarity (Å²).","Hydrogen Bond Donors":"Count of groups able to donate hydrogen bonds.","Hydrogen Bond Acceptors":"Count of atoms/groups able to accept hydrogen bonds.","Rotatable Bonds":"Approximate count of rotatable single bonds; a rough flexibility indicator.","Ring Count":"Number of rings in the molecular graph.","Aromatic Rings":"Count of aromatic rings.","Aliphatic Rings":"Count of non-aromatic rings.","Fraction Csp3":"Fraction of carbon atoms with sp³ hybridization.","Molar Refractivity":"Descriptor related to molecular volume and polarizability.","Heavy Atoms":"All atoms except hydrogen.","Number of Atoms":"Atoms explicitly present in the parsed structure."
}

def parse_mol(smiles):
    if not smiles or not smiles.strip(): return None
    try: return Chem.MolFromSmiles(smiles.strip())
    except Exception: return None

def properties(m):
    return {
      "Molecular Formula":rdMolDescriptors.CalcMolFormula(m),
      "Molecular Weight":round(Descriptors.MolWt(m),3),
      "Exact Molecular Weight":round(Descriptors.ExactMolWt(m),4),
      "LogP":round(Crippen.MolLogP(m),3),
      "TPSA":round(rdMolDescriptors.CalcTPSA(m),3),
      "Hydrogen Bond Donors":Lipinski.NumHDonors(m),
      "Hydrogen Bond Acceptors":Lipinski.NumHAcceptors(m),
      "Rotatable Bonds":Lipinski.NumRotatableBonds(m),
      "Ring Count":rdMolDescriptors.CalcRingCount(m),
      "Aromatic Rings":rdMolDescriptors.CalcNumAromaticRings(m),
      "Aliphatic Rings":rdMolDescriptors.CalcNumAliphaticRings(m),
      "Number of Atoms":m.GetNumAtoms(),
      "Heavy Atoms":m.GetNumHeavyAtoms(),
      "Fraction Csp3":round(rdMolDescriptors.CalcFractionCSP3(m),3),
      "Molar Refractivity":round(Crippen.MolMR(m),3)
    }

def molecule_image(m, size=(620,400)):
    cp=Chem.Mol(m); Chem.rdDepictor.Compute2DCoords(cp)
    return Draw.MolToImage(cp,size=size, kekulize=True, fitImage=True, options=None)

def make_3d(m, style="stick", spin=True, surface=False):
    mol3d=Chem.AddHs(Chem.Mol(m))
    params=AllChem.ETKDGv3(); params.randomSeed=42
    if AllChem.EmbedMolecule(mol3d,params)!=0: return None
    try: AllChem.MMFFOptimizeMolecule(mol3d,maxIters=150)
    except Exception:
        try: AllChem.UFFOptimizeMolecule(mol3d,maxIters=150)
        except Exception: pass
    block=Chem.MolToMolBlock(mol3d)
    view=py3Dmol.view(width=850,height=490)
    view.addModel(block,"mol")
    if style=="Ball & stick": view.setStyle({"stick":{"radius":0.16},"sphere":{"scale":0.29}})
    elif style=="Space filling": view.setStyle({"sphere":{"scale":0.78}})
    elif style=="Wireframe": view.setStyle({"line":{}})
    elif style=="Stick": view.setStyle({"stick":{"radius":0.18}})
    else: view.setStyle({"stick":{"radius":0.13},"sphere":{"scale":0.25}})
    if surface:
        try: view.addSurface(py3Dmol.VDW, {"opacity":0.22,"color":"#66e6df"})
        except Exception: pass
    view.setBackgroundColor("#0b1728")
    view.zoomTo()
    if spin: view.spin(True)
    return view._make_html()

def pdf_report(name, reg, smiles, m, props):
    b=BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42)
    stys=getSampleStyleSheet(); stys.add(ParagraphStyle(name='Sub',parent=stys['Heading2'],textColor=colors.HexColor('#167e83'),spaceBefore=12))
    story=[Paragraph('MOLECULE STUDIO',stys['Title']),Paragraph('CHEMINFORMATICS ANALYSIS REPORT',stys['Heading2']),Spacer(1,10)]
    meta=[["Student",name or "Not provided"],["Registration number",reg or "Not provided"],["Input SMILES",smiles],["Canonical SMILES",Chem.MolToSmiles(m)],["Molecular formula",props['Molecular Formula']]]
    t=Table(meta,colWidths=[1.7*inch,4.8*inch]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.lightgrey),('BACKGROUND',(0,0),(0,-1),colors.HexColor('#e8f7f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('PADDING',(0,0),(-1,-1),7)])); story += [t,Spacer(1,14)]
    story.append(Paragraph('Molecular descriptors',stys['Sub']))
    data=[["Descriptor","Value","Interpretation"]]
    for k,v in props.items(): data.append([k,str(v),PROPERTY_HELP.get(k,"Calculated molecular descriptor.")])
    t=Table(data,colWidths=[1.6*inch,1.05*inch,3.85*inch],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#122b42')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.35,colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP'),('FONTSIZE',(0,0),(-1,-1),8),('LEADING',(0,0),(-1,-1),10),('PADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f9fb')])]))
    story += [t,Spacer(1,12),Paragraph('Interpretation note',stys['Sub']),Paragraph('Descriptors are computational estimates derived from the submitted molecular graph. They support learning and comparison and should not be treated as experimental measurements or as a substitute for expert chemical review.',stys['BodyText'])]
    doc.build(story); return b.getvalue()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("<div style='font-size:25px;font-weight:800;letter-spacing:-1px'>⚛️ MOLECULE<span style='color:#65e7df'>.STUDIO</span></div><div class='small-note'>CHEMINFORMATICS • VIRTUAL LAB</div>",unsafe_allow_html=True)
    st.divider()
    page=st.radio("WORKSPACE",["Overview","Molecular explorer","Property analysis","Compare molecules","Learn chemistry","Knowledge check","Export report"],label_visibility="visible")
    st.divider()
    st.markdown("<div class='section-kicker'>SESSION PROFILE</div>",unsafe_allow_html=True)
    student=st.text_input("Student name",key="student_name",placeholder="Your name")
    reg=st.text_input("Registration number",key="reg_number",placeholder="Optional")
    st.markdown("<div class='small-note'>Built for curious minds. Explore molecular structure, descriptors and structure–property relationships.</div>",unsafe_allow_html=True)

if "smiles" not in st.session_state: st.session_state.smiles=EXAMPLES["Caffeine"]["smiles"]
if "mol_name" not in st.session_state: st.session_state.mol_name="Caffeine"

def hero(kicker,title,desc):
    st.markdown(f"<div class='hero'><div class='pill'>✦ {kicker}</div><h1>{title}</h1><p>{desc}</p></div>",unsafe_allow_html=True)
def metric(label,value,unit=""):
    st.markdown(f"<div class='metric'><div class='label'>{label}</div><div class='value'>{value} <span class='unit'>{unit}</span></div></div>",unsafe_allow_html=True)
def choose_molecule(key="mol_select"):
    cols=st.columns([1,2,1])
    with cols[0]:
        picked=st.selectbox("Start with a molecule",list(EXAMPLES.keys()),index=list(EXAMPLES.keys()).index(st.session_state.mol_name) if st.session_state.mol_name in EXAMPLES else 0,key=key)
    with cols[1]:
        entered=st.text_input("SMILES notation",value=st.session_state.smiles,key=key+"_smiles",help="Paste a SMILES string to describe molecular connectivity.")
    with cols[2]:
        st.write(""); st.write("")
        if st.button("Load example",key=key+"_load",use_container_width=True):
            st.session_state.mol_name=picked; st.session_state.smiles=EXAMPLES[picked]["smiles"]; st.rerun()
    return entered,picked

# ---------- overview ----------
if page=="Overview":
    hero("COMPUTATIONAL CHEMISTRY, REIMAGINED","Molecules, made visible.","An interactive cheminformatics workspace to translate molecular notation into visual structures, explore physicochemical properties and understand how structure shapes behaviour.")
    st.write("")
    cols=st.columns(4)
    for c,(n,t,s) in zip(cols,[("01","Visualize","2D structures + animated 3D models"),("02","Characterize","15+ calculated descriptors"),("03","Compare","Spot differences across molecules"),("04","Discover","Guided theory and quick quizzes")]):
        with c: st.markdown(f"<div class='panel' style='min-height:145px'><div class='section-kicker'>{n} / LAB MODULE</div><h3 style='margin:.2rem 0'>{t}</h3><div class='small-note'>{s}</div></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>QUICK START</div><h2>Choose a molecule to begin</h2>",unsafe_allow_html=True)
    cols=st.columns(4)
    for c,name in zip(cols,list(EXAMPLES.keys())[:4]):
        with c:
            st.markdown(f"<div class='panel'><div style='font-size:24px;color:#65e7df'>◉</div><b>{name}</b><div class='small-note' style='margin-top:6px'>{EXAMPLES[name]['note']}</div></div>",unsafe_allow_html=True)
            if st.button(f"Explore {name}",key="quick_"+name,use_container_width=True): st.session_state.mol_name=name; st.session_state.smiles=EXAMPLES[name]["smiles"]; st.session_state.page_hint="Molecular explorer"; st.rerun()
    st.markdown("<div class='panel'><div class='section-kicker'>YOUR LAB NOTEBOOK</div><h3>From a line of text to a molecule you can inspect</h3><div class='small-note'>SMILES encodes atoms, bonds, branches and rings in a compact string. Molecule Studio parses that representation, computes molecular descriptors with RDKit, and generates an interactive 3D conformer for exploration. Results are computational and educational—not experimental measurements.</div></div>",unsafe_allow_html=True)

elif page=="Molecular explorer":
    hero("LAB 01 / STRUCTURE","Molecular explorer","Enter a SMILES string, inspect its 2D connectivity and rotate a generated 3D conformer. Switch between common molecular rendering styles.")
    smiles,picked=choose_molecule("explorer")
    m=parse_mol(smiles)
    if not m: st.error("This SMILES string could not be parsed. Check atom symbols, brackets, bond markers and ring numbers.")
    else:
        p=properties(m); st.success("Valid molecular structure parsed")
        c1,c2=st.columns([1.1,1])
        with c1:
            st.markdown("<div class='section-kicker'>2D CONNECTIVITY</div>",unsafe_allow_html=True)
            st.image(molecule_image(m),use_container_width=True)
        with c2:
            st.markdown("<div class='section-kicker'>MOLECULE IDENTITY</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='panel'><div class='small-note'>MOLECULAR FORMULA</div><div style='font-size:2rem;font-weight:800;color:#65e7df'>{p['Molecular Formula']}</div><div class='small-note' style='margin-top:10px'>Canonical SMILES</div><code>{Chem.MolToSmiles(m)}</code><hr><div class='small-note'>Atoms · {m.GetNumAtoms()} &nbsp;&nbsp; Heavy atoms · {m.GetNumHeavyAtoms()} &nbsp;&nbsp; Rings · {p['Ring Count']}</div></div>",unsafe_allow_html=True)
            st.caption("Tip: use lowercase atom symbols for aromatic atoms in SMILES (for example, `c1ccccc1` for benzene).")
        st.markdown("<div class='section-kicker'>INTERACTIVE 3D CONFORMER</div><h2>Rotate. Inspect. Understand.</h2>",unsafe_allow_html=True)
        a,b,c=st.columns([1,1,1])
        with a: style=st.selectbox("Rendering style",["Ball & stick","Stick","Space filling","Wireframe"],index=0)
        with b: spin=st.toggle("Auto-rotate model",value=True)
        with c: surface=st.toggle("Van der Waals surface",value=False)
        if st.button("Generate 3D model",key="make_3d",use_container_width=True):
            with st.spinner("Embedding and optimizing a 3D conformer…"):
                html=make_3d(m,style,spin,surface)
            if html: components.html(html,height=510,scrolling=False)
            else: st.error("A 3D conformer could not be generated for this structure.")
        else:
            st.markdown("<div class='panel'><div class='small-note'>Select your preferred rendering and generate the 3D model. Drag to rotate, scroll to zoom. Auto-rotation is enabled by default.</div></div>",unsafe_allow_html=True)
        st.markdown("<div class='section-kicker'>KEY DESCRIPTORS</div>",unsafe_allow_html=True)
        q=st.columns(4)
        for col,(lab,val,unit) in zip(q,[("Molecular weight",p['Molecular Weight'],"g/mol"),("LogP",p['LogP'],""),("Polar surface area",p['TPSA'],"Å²"),("H-bond donors",p['Hydrogen Bond Donors'],"")]):
            with col: metric(lab,val,unit)

elif page=="Property analysis":
    hero("LAB 02 / DESCRIPTORS","Property analysis","Read the molecular fingerprint through calculated descriptors, from molecular mass and polarity to hydrogen bonding, ring systems and flexibility.")
    smiles,picked=choose_molecule("props")
    m=parse_mol(smiles)
    if not m: st.error("Please enter a valid SMILES string.")
    else:
        p=properties(m)
        st.markdown(f"<div class='panel'><div class='small-note'>ANALYSIS TARGET</div><h2 style='margin:.2rem 0'>{picked if smiles==EXAMPLES[picked]['smiles'] else 'Custom molecule'} · {p['Molecular Formula']}</h2><div class='small-note'>Canonical SMILES: <code>{Chem.MolToSmiles(m)}</code></div></div>",unsafe_allow_html=True)
        cols=st.columns(4)
        for col,(lab,val,unit) in zip(cols,[("Molecular weight",p['Molecular Weight'],"g/mol"),("LogP",p['LogP'],""),("TPSA",p['TPSA'],"Å²"),("Ring count",p['Ring Count'],"")]):
            with col: metric(lab,val,unit)
        cols=st.columns(4)
        for col,(lab,val,unit) in zip(cols,[("H-bond donors",p['Hydrogen Bond Donors'],""),("H-bond acceptors",p['Hydrogen Bond Acceptors'],""),("Rotatable bonds",p['Rotatable Bonds'],""),("Fraction Csp³",p['Fraction Csp3'],"")]):
            with col: metric(lab,val,unit)
        st.markdown("<div class='section-kicker'>FULL DESCRIPTOR TABLE</div>",unsafe_allow_html=True)
        df=pd.DataFrame([{"Descriptor":k,"Value":v,"What it tells you":PROPERTY_HELP.get(k,"Calculated molecular descriptor.")} for k,v in p.items()])
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download descriptor table (CSV)",df.to_csv(index=False).encode(),file_name="molecule_descriptors.csv",mime="text/csv")
        with st.expander("How should I interpret these values?"):
            st.markdown("- **Molecular weight** describes the molecule's mass.\n- **LogP** estimates preference for non-polar versus aqueous environments; it is not a direct solubility measurement.\n- **TPSA** summarizes polar surface contributions.\n- **H-bond donors/acceptors** count likely interaction sites.\n- **Rotatable bonds** offer a rough view of flexibility.\n- **Fraction Csp³** gives a simple measure of saturated 3D carbon character.\n\nDescriptors are best interpreted together and do not independently predict biological activity.")

elif page=="Compare molecules":
    hero("LAB 03 / COMPARISON","Structure–property comparison","Compare several molecules side by side and see how their structural differences appear in descriptor values.")
    default="Ethanol,CCO\nBenzene,c1ccccc1\nPhenol,Oc1ccccc1\nAspirin,CC(=O)Oc1ccccc1C(=O)O\nCaffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    raw=st.text_area("One molecule per line: name, SMILES",value=default,height=180)
    rows=[]; errors=[]
    for line in raw.splitlines():
        if not line.strip(): continue
        parts=line.split(',',1)
        if len(parts)!=2: errors.append(line); continue
        name,smi=parts[0].strip(),parts[1].strip(); m=parse_mol(smi)
        if m:
            p=properties(m); rows.append({"Molecule":name,"Formula":p["Molecular Formula"],"Molecular weight":p["Molecular Weight"],"LogP":p["LogP"],"TPSA":p["TPSA"],"HBD":p["Hydrogen Bond Donors"],"HBA":p["Hydrogen Bond Acceptors"],"Rotatable bonds":p["Rotatable Bonds"],"Rings":p["Ring Count"],"SMILES":smi})
        else: errors.append(f"{name}: invalid SMILES")
    if errors: st.warning("Could not read: "+"; ".join(errors))
    if rows:
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        chart_metric=st.selectbox("Visualize descriptor",["Molecular weight","LogP","TPSA","HBD","HBA","Rotatable bonds","Rings"])
        st.bar_chart(df.set_index("Molecule")[[chart_metric]],use_container_width=True)
        st.download_button("Download comparison (CSV)",df.to_csv(index=False).encode(),file_name="molecule_comparison.csv",mime="text/csv")
        st.caption("Comparisons are calculated from each submitted SMILES. The values are descriptors, not experimental assay results.")

elif page=="Learn chemistry":
    hero("FIELD NOTES / THEORY","Chemistry behind the interface","A quick reference to the concepts that connect molecular notation, structure and computed properties.")
    with st.expander("01 · What is cheminformatics?",expanded=True):
        st.write("Cheminformatics applies computation and data science to chemical information. It helps represent molecules, calculate descriptors, organize chemical libraries and explore relationships between molecular structure and properties.")
    with st.expander("02 · SMILES notation"):
        st.write("SMILES (Simplified Molecular Input Line Entry System) represents a molecular graph as text. Atoms are letters, bond types may be explicit, branches use parentheses, and ring closures use matching digits. Aromatic atoms are commonly written in lowercase.")
        st.code("Ethanol   CCO\nBenzene   c1ccccc1\nAcetic acid   CC(=O)O\nAspirin   CC(=O)Oc1ccccc1C(=O)O",language="text")
    with st.expander("03 · Reading 3D molecular models"):
        st.write("Ball-and-stick emphasizes connectivity and approximate geometry. Stick views make bonding patterns prominent. Space-filling views approximate occupied volume. Wireframe is a minimal representation. A generated conformer is one plausible geometry, not necessarily the only low-energy form in solution.")
    with st.expander("04 · Structure–property relationships"):
        st.write("Molecular properties emerge from atom types, connectivity, functional groups, shape and flexibility. Descriptors simplify some of these features into numbers. Use multiple descriptors together and remember that real-world behaviour can depend on solvent, temperature, pH, conformation and experimental conditions.")
    with st.expander("05 · Sigma and pi bonds"):
        st.write("A sigma bond is formed by end-on orbital overlap and is the first bond between two atoms. Additional components in double and triple bonds are pi bonds formed by sideways overlap. Aromatic bonding is delocalized, so simple pi-bond counting conventions should be interpreted carefully.")
    st.markdown("<div class='panel'><div class='section-kicker'>LEARNING PATH</div><b>Notation → Connectivity → Geometry → Descriptors → Comparison</b><div class='small-note' style='margin-top:8px'>Move through the workflow in this order to connect what a molecule looks like with what its calculated properties suggest.</div></div>",unsafe_allow_html=True)

elif page=="Knowledge check":
    hero("LAB 04 / SELF-CHECK","Knowledge check","Test the basics. Choose an answer for each question and submit to review your score and explanations.")
    questions=[("What does SMILES primarily represent?",["A measured boiling point","A molecular structure as text","A 3D animation file","A reaction rate"],1,"SMILES encodes molecular connectivity in a text string."),("Which descriptor estimates lipophilicity?",["TPSA","Ring count","LogP","Molecular formula"],2,"LogP is a commonly used estimate of the octanol/water partition coefficient."),("What does TPSA describe?",["Topological polar surface area","Total pressure at standard atmosphere","Number of pi bonds","Exact molecular mass"],0,"TPSA is a descriptor associated with molecular polarity."),("Which view best emphasizes approximate occupied volume?",["Wireframe","Space filling","SMILES text","2D labels only"],1,"Space-filling models use atom spheres to represent approximate van der Waals volume."),("A generated 3D conformer should be treated as…",["The only possible shape","An experimental measurement","One plausible geometry for exploration","Proof of biological activity"],2,"Molecules can adopt multiple conformations; a generated conformer is not experimental proof.")]
    answers=[]
    for i,(q,opts,correct,explain) in enumerate(questions): answers.append(st.radio(f"{i+1:02d}  {q}",opts,key=f"quiz_{i}",index=None))
    if st.button("Submit assessment",use_container_width=True):
        score=0
        for i,((q,opts,correct,explain),ans) in enumerate(zip(questions,answers)):
            if ans==opts[correct]: score+=1; st.success(f"{i+1}. Correct — {explain}")
            else: st.error(f"{i+1}. Correct answer: {opts[correct]}. {explain}")
        st.markdown(f"<div class='panel'><div class='section-kicker'>YOUR RESULT</div><div style='font-size:2.5rem;font-weight:800;color:#65e7df'>{score}/{len(questions)}</div><div class='small-note'>{round(score/len(questions)*100)}% · {'Excellent work!' if score>=4 else 'Review the field notes and try again.'}</div></div>",unsafe_allow_html=True)

elif page=="Export report":
    hero("LAB NOTEBOOK / EXPORT","Your molecular report","Generate a portable report with the molecule identity, calculated descriptors and brief interpretation notes.")
    smiles,picked=choose_molecule("report")
    m=parse_mol(smiles)
    if not m: st.error("Enter a valid SMILES before generating a report.")
    else:
        p=properties(m)
        st.markdown(f"<div class='panel'><div class='small-note'>REPORT PREVIEW</div><h2>{p['Molecular Formula']} · {picked if smiles==EXAMPLES[picked]['smiles'] else 'Custom molecule'}</h2><div class='small-note'>Canonical SMILES: <code>{Chem.MolToSmiles(m)}</code></div></div>",unsafe_allow_html=True)
        st.image(molecule_image(m,(500,300)),width=500)
        st.dataframe(pd.DataFrame([{"Descriptor":k,"Value":v,"Interpretation":PROPERTY_HELP.get(k,"Calculated descriptor.")} for k,v in p.items()]),use_container_width=True,hide_index=True)
        st.download_button("Download CSV data",pd.DataFrame([{"Descriptor":k,"Value":v,"Interpretation":PROPERTY_HELP.get(k,"Calculated descriptor.")} for k,v in p.items()]).to_csv(index=False).encode(),file_name="molecule_report.csv",mime="text/csv")
        st.download_button("Generate & download PDF report",pdf_report(student,reg,smiles,m,p),file_name="Molecule_Studio_Report.pdf",mime="application/pdf",use_container_width=True)
        st.caption("Report includes student details when provided in the sidebar. Leave registration number blank if you prefer not to include it.")

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;color:#6f88a6;font:10px DM Mono,monospace;letter-spacing:.08em'><span>MOLECULE.STUDIO · CHEMINFORMATICS VIRTUAL LAB</span><span>EDUCATIONAL COMPUTATIONAL TOOL · NOT EXPERIMENTAL DATA</span></div>",unsafe_allow_html=True)
