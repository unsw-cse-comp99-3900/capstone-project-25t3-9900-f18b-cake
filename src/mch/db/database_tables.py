import yaml

from typing import Optional
from datetime import datetime

from sqlmodel import Field, SQLModel


class AnalysisSet(SQLModel, table=True):
    __tablename__ = "zcc_analysis_set"
    analysis_set_id: Optional[str] = Field(default=None, primary_key=True)
    patient_id: str
    public_subject_id: int

class Purity(SQLModel, table=True):
    __tablename__ = "zcc_purity"
    purity_id: Optional[int] = Field(default=None, primary_key=True)
    analysis_set_id: str
    purity: float

class AnalysisSetXRef(SQLModel, table=True):
    __tablename__ = "zcc_analysis_set_biosample_xref"
    biosample_id: Optional[str] = Field(default=None, primary_key=True)
    analysis_set_id: str
    
class Sample(SQLModel, table=True):
    __tablename__ = "zcc_sample" 
    sample_id: Optional[str] = Field(default=None, primary_key=True)
    meth_sample_id: str
    meth_id: str
    rnaseq_id: Optional[str] = Field(default=None, foreign_key="zcc_curated_somatic_rnaseq_counts.rnaseq_id")
    #rnaseq_id: str
    study: str
    zcc_sample_id: str
    patient_id: str
    cancer_category: str
    cancer_type: str
    cancer_subtype: str
    diagnosis: str
    histologic_diagnosis: str
    final_diagnosis: str
    zero2_category: str
    zero2_subcategory1: str
    zero2_subcategory2: str
    zero2_final_diagnosis: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
class BioSample(SQLModel, table=True):
    __tablename__ = "zcc_biosample"
    biosample_id: str = Field(primary_key=True)
    patient_id: str
    #zcc_sample_id: str
    public_subject_id: int
    sample_type: str
    biosample_status: str
    biosample_type: str

class MethylationPredictions(SQLModel, table=True):
    __tablename__  = "zcc_methylation_predictions"
    sample_id: Optional[str] = Field(default=None, primary_key=True)
    meth_group_id: Optional[str] = Field(default=None, primary_key=True)
    meth_class_score:float

class MethylationGroup(SQLModel, table=True):
    __tablename__ = "zcc_methylation_group"
    meth_class_id: Optional[str] = Field(default=None, primary_key=True)
    meth_class: str
    meth_classifier_id: str
    meth_group_id: str
    group_name: str

class MethylationClassifier(SQLModel, table=True):
    __tablename__  = "zcc_methylation_classifier"
    meth_classifier_id: Optional[str] = Field(default=None, primary_key=True)
    name: str
    version: str
    description: str

    
class CuratedMethylation(SQLModel, table=True):
    __tablename__ = "zcc_curated_sample_somatic_methylation"   
    sample_id:  Optional[str] = Field(default=None, primary_key=True)
    meth_group_id:  Optional[str] = Field(default=None, primary_key=True)
    meth_class_score: float
    interpretation: str
    match_zcc: int
    classification: str
    reportable: int
    in_molecular_report: int
    targetable: int
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    updated_at:  datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    updated_by: str
    
class SampleSomaticSnvs(SQLModel, table=True):
    __tablename__ = "zcc_curated_sample_somatic_snv"
    sample_id: Optional[str] = Field(default=None, primary_key=True)
    variant_id: str
    
class SomaticSnvs(SQLModel, table=True):
    __tablename__ = "zcc_curated_snv"
    variant_id: Optional[str] = Field(default=None, primary_key=True)
    gene_id: Optional[int] = Field(default=None, foreign_key="zcc_genes.gene_id")

class Gene(SQLModel, table=True):
    __tablename__ = "zcc_genes"
    gene_id: Optional[int] = Field(default=None, primary_key=True)
    gene: str
    chromosome: str
    gene_start: int
    gene_end: int

class RNASeqCounts(SQLModel, table=True):
    __tablename__ = "zcc_curated_sample_somatic_rnaseq"
    rnaseq_id: Optional[str] = Field(default=None, primary_key=True)
    gene_id: Optional[int] = Field(default=None, foreign_key="zcc_genes.gene_id")
    fpkm: float
    logFC: float
    pvalue: float
    zscore_mean: float
    fpkm_mean: float


    
    
