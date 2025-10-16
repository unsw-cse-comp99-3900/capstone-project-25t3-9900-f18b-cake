
from typedb.client import *

def typedbQuery(query: str, inference: bool) -> list:
    result = []
    with TypeDB.core_client("localhost:1729") as typeDBclient:
        with typeDBclient.session("oncotree", SessionType.DATA) as session:
            options = TypeDBOptions.core()
            options.infer = inference
            oncotreeResults = getMatch(query, session, options)
            for oncotreeResult in oncotreeResults:
                result.append(oncotreeResult)
        pass
    return result


def getMatch(query: str, session, options) -> dict:
        read_transaction = session.transaction(TransactionType.READ, options) # use your typedb connection session
        answer_iterator = read_transaction.query().match(query)
        result = []
        for ans in answer_iterator:
            item = {k: v.get_value() if v.is_attribute() else v for k, v in ans.map().items()}
            result.append(item)
        read_transaction.close()

        return result