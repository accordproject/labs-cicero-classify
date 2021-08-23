const ModelManager = require('@accordproject/concerto-core').ModelManager;
const Concerto = require('@accordproject/concerto-core').Concerto;
const modelManager = new ModelManager();
modelManager.addModelFile( `namespace org.acme.address
concept PostalAddress {
  o String streetAddress optional
  o String postalCode optional
  o String postOfficeBoxNumber optional
  o String addressRegion optional
  o String addressLocality optional
  o String addressCountry optional
}`, 'model.cto');

const postalAddress = {
    $class : 'org.acme.address.PostalAddress',
    streetAddress : '1 Maine Street',
    postalCode: "9487"
};

// const postalAddress = {
//     $class : 'org.acme.address.PostalAddress',
//     streetAddress : '1 Maine Street',
//     AAAAA : '1 Maine Street'
// };

const concerto = new Concerto(modelManager);
concerto.validate(postalAddress);

const typeDeclaration = concerto.getTypeDeclaration(postalAddress);
const fqn = typeDeclaration.getFullyQualifiedName();
console.log(typeDeclaration); // should equal 'org.acme.address.PostalAddress'
